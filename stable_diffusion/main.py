import argparse
import datetime
import glob
import os
import sys
import time
import random

import numpy as np
import torch
import torchvision

import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.utilities import rank_zero_info

LIGHTNING_PACK_NAME = "lightning.pytorch."

from functools import partial

from omegaconf import OmegaConf
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader, Dataset

from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config

import mlperf_logging_utils
import mlperf_logging.mllog.constants as mllog_constants
from mlperf_logging_utils import mllogger


def get_parser(**parser_kwargs):
    # A function to create an ArgumentParser object and add arguments to it

    def str2bool(v):
        # A helper function to parse boolean values from command line arguments
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")
    # Create an ArgumentParser object with specifies kwargs
    parser = argparse.ArgumentParser(**parser_kwargs)

    # Add vairous command line arguments with their default balues and descriptions
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="train",
        choices=["train", "validate"],
        help="run mode, train or validation",
    )
    parser.add_argument(
        "-v",
        "--validation",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="validation",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project",
    )
    parser.add_argument(
        "-c",
        "--ckpt",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="load pretrained checkpoint from stable AI",
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=random.SystemRandom().randint(0, 2**32 - 1),
        help="seed for seed_everything",
    )
    parser.add_argument(
        "--fid_threshold",
        type=int,
        default=90,
        help="halt training once this FID validation score or a smaller one is achieved."
             "if used with --clip_threshold, both metrics need to reach their targets.",
    )
    parser.add_argument(
        "--clip_threshold",
        type=int,
        default=0.15,
        help="halt training once this CLIP validation score or a higher one is achieved."
             "if used with --clip_threshold, both metrics need to reach their targets.",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="/tmp/stable_diffusion/results",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    parser.add_argument(
        "--train_log_interval",
        type=int,
        default=100,
        help="Training logging interval"
    )
    parser.add_argument(
        "--validation_log_interval",
        type=int,
        default=10,
        help="Validation logging interval"
    )

    return parser

# A function that returns the non-default arguments between two objects
def nondefault_trainer_args(opt):
    # create an argument parsser
    parser = argparse.ArgumentParser()
    # add pytorch lightning trainer default arguments
    parser = Trainer.add_argparse_args(parser)
    # parse the empty arguments to obtain the default values
    args = parser.parse_args([])
    # return all non-default arguments
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))

class ListEarlyStopping(Callback):
    # Early stopping class that accepts a list of metrics and all stopping_thresholds
    # must be met before stopping
    def __init__(self, monitor_metrics: list = ["validation/fid", "validation/clip"],
                 mode_metrics: dict = {'validation/fid': 'min', 'validation/clip': 'max'},
                 stopping_thresholds: dict = {"validation/fid": None, "validation/clip": None},
                 check_finite: bool = False):
        super(ListEarlyStopping, self).__init__()

        self.monitor_metrics = monitor_metrics
        self.mode_metrics = mode_metrics
        self.stopping_thresholds = stopping_thresholds
        self.check_finite = check_finite

    def check_metrics(self, current_metrics):
        should_stop = []
        for metric in self.monitor_metrics:
            if metric in current_metrics:
                current_value = current_metrics[metric]

                if self.check_finite and not torch.isfinite(torch.as_tensor(current_value)):
                    raise ValueError(f"The monitored metric {metric} has become non-finite.")

                # Skip metrics without a stopping thresholds
                if self.stopping_thresholds[metric] is None:
                    continue

                if self.mode_metrics[metric] == 'min':
                    should_stop.append(current_value <= self.stopping_thresholds[metric])

                if self.mode_metrics[metric] == 'max':
                    should_stop.append(current_value >= self.stopping_thresholds[metric])

        # A minimum of one metric should have been reviewed.
        return False if not should_stop else all(should_stop)

    def on_validation_end(self, trainer, pl_module):
        logs = trainer.callback_metrics
        should_stop = self.check_metrics(logs)
        if should_stop:
            rank_zero_info('Early stopping conditioned have been met. Stopping training.')
            trainer.should_stop = True

class SetupCallback(Callback):
    # I nitialize the callback with the necessary parameters

    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    # Save a checkpoint if training is interrupted with keyboard interrupt
    def on_keyboard_interrupt(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    # Create necessary directories and save configuration files before training starts
    # def on_pretrain_routine_start(self, trainer, pl_module):
    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config, os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            # Save project config and lightning config as YAML files
            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        # Remove log directory if resuming training and directory already exists
        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass

    # def on_fit_end(self, trainer, pl_module):
    #     if trainer.global_rank == 0:
    #         ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
    #         rank_zero_info(f"Saving final checkpoint in {ckpt_path}.")
    #         trainer.save_checkpoint(ckpt_path)


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py

    def on_train_start(self, trainer, pl_module):
        rank_zero_info("Training is starting")

    # the method is called at the end of each training epoch
    def on_train_end(self, trainer, pl_module):
        rank_zero_info("Training is ending")

    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.strategy.root_device.index)
        torch.cuda.synchronize(trainer.strategy.root_device.index)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        torch.cuda.synchronize(trainer.strategy.root_device.index)
        max_memory = torch.cuda.max_memory_allocated(trainer.strategy.root_device.index) / 2**20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.strategy.reduce(max_memory)
            epoch_time = trainer.strategy.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass


class DummyDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=4, image_size=64, num_samples=100):
        super().__init__()
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_samples = num_samples
        self.device = 'cpu'  # Always create on CPU, Lightning will handle device movement
        print("Batch size", self.batch_size)

    def setup(self, stage=None):
        # Generate random image and text data
        self.train_images = torch.randn(self.num_samples, 3, self.image_size, self.image_size, device=self.device, dtype=torch.float32)
        self.train_texts = [f"Sample text {i}" for i in range(self.num_samples)]
        self.val_images = torch.randn(self.num_samples // 10, 3, self.image_size, self.image_size, device=self.device, dtype=torch.float32)
        self.val_texts = [f"Val text {i}" for i in range(self.num_samples // 10)]

    def train_dataloader(self):
        dataset = [{'npy': img, 'txt': txt, 'caption': txt} for img, txt in zip(self.train_images, self.train_texts)]
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self._collate_fn)

    def val_dataloader(self):
        dataset = [{'npy': img, 'txt': txt, 'caption': txt, 'image_id': f"dummy_{i}", 'id': i} 
                  for i, (img, txt) in enumerate(zip(self.val_images, self.val_texts))]
        return DataLoader(dataset, batch_size=self.batch_size, collate_fn=self._collate_fn)

    def _collate_fn(self, batch):
        # Collate function to create batches in the correct format
        collated = {
            'npy': torch.stack([item['npy'] for item in batch]).float(),  # Ensure float32
            'txt': [item['txt'] for item in batch],
            'caption': [item['caption'] for item in batch],
        }
        # Add validation-specific fields if they exist
        if 'image_id' in batch[0]:
            collated.update({
                'image_id': [item['image_id'] for item in batch],
                'id': [item['id'] for item in batch],
            })
        return collated

    def on_before_batch_transfer(self, batch, dataloader_idx):
        # Ensure tensors are on CPU before Lightning moves them
        if isinstance(batch, dict) and 'npy' in batch:
            batch['npy'] = batch['npy'].to('cpu')
        return batch


if __name__ == "__main__":
    # custom parser to specify config files, train, test and debug mode,
    # postfix, resume.
    # `--key value` arguments are interpreted as arguments to the trainer.
    # `nested.key=value` arguments are interpreted as config parameters.
    # configs are merged from left-to-right followed by command line parameters.

    # model:
    #   base_learning_rate: float
    #   target: path to lightning module
    #   params:
    #       key: value
    # data:
    #   target: main.DataModuleFromConfig
    #   params:
    #      batch_size: int
    #      wrap: bool
    #      train:
    #          target: path to train dataset
    #          params:
    #              key: value
    #      validation:
    #          target: path to validation dataset
    #          params:
    #              key: value
    #      test:
    #          target: path to test dataset
    #          params:
    #              key: value
    # lightning: (optional, has sane defaults and can be specified on cmdline)
    #   trainer:
    #       additional arguments to trainer
    #   logger:
    #       logger to instantiate
    #   modelcheckpoint:
    #       modelcheckpoint to instantiate
    #   callbacks:
    #       callback1:
    #           target: importpath
    #           params:
    #               key: value

    # Setup mllogger
    # mllogger = mllog.get_mllogger()
    mlperf_logging_utils.submission_info(mllogger=mllogger,
                                         submission_benchmark=mllog_constants.STABLE_DIFFUSION,
                                         submission_division=mllog_constants.CLOSED,
                                         submission_org="reference_implementation",
                                         submission_platform="DGX-A100",
                                         submission_poc_name="Ahmad Kiswani",
                                         submission_poc_email="akiswani@nvidia.com",
                                         submission_status=mllog_constants.ONPREM)

    mllogger.start(key=mllog_constants.INIT_START)

    # get the current time to create a new logging directory
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    status = mllog_constants.ABORTED
    
    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()
    # Veirfy the arguments are both specified
    if opt.name and opt.resume:
        raise ValueError("-n/--name and -r/--resume cannot be specified both."
                         "If you want to resume training in a new log folder, "
                         "use -n/--name in combination with --resume_from_checkpoint")

    # Check if the "resume" option is specified, resume training from the checkpoint if it is true
    ckpt = None
    if opt.resume:
        rank_zero_info("Resuming from {}".format(opt.resume))
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            # idx = len(paths)-paths[::-1].index("logs")+1
            # logdir = "/".join(paths[:idx])
            logdir = "/".join(paths[:-2])
            rank_zero_info("logdir: {}".format(logdir))
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        # Finds all ".yaml" configuration files in the log directory and adds them to the list of base configurations
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        # Gets the name of the current log directory by splitting the path and taking the last element.
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            rank_zero_info("Using base config {}".format(opt.base))
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)

        # Sets the checkpoint path of the 'ckpt' option is specified
        if opt.ckpt:
            ckpt = opt.ckpt

    # Create the checkpoint and configuration directories within the log directory.
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    # Sets the seed for the random number generator to ensure reproducibility
    mllogger.event(key=mllog_constants.SEED, value=opt.seed)
    seed_everything(opt.seed)

    # Intinalize and save configuratioon using the OmegaConf library.
    try:
        trainer = None  # Initialize trainer to None at start
        # init and save configs
        if len(opt.base) == 0:
            rank_zero_info("No base config specified. Using default configuration from train_01x08x08.yaml")
            default_config_path = os.path.join(os.path.dirname(__file__), "configs", "train_01x08x08.yaml")
            if os.path.exists(default_config_path):
                config = OmegaConf.load(default_config_path)
                
                # Update paths to use /tmp
                if "validation_config" in config.model.params:
                    if "save_images" in config.model.params.validation_config:
                        config.model.params.validation_config.save_images.base_output_dir = "/tmp/stable_diffusion/results"
                    if "fid" in config.model.params.validation_config:
                        config.model.params.validation_config.fid.cache_dir = "/tmp/stable_diffusion/checkpoints/inception"
                        config.model.params.validation_config.fid.gt_path = "/tmp/stable_diffusion/datasets/coco2014/val2014_30k_stats.npz"
                    if "clip" in config.model.params.validation_config:
                        config.model.params.validation_config.clip.cache_dir = "/tmp/stable_diffusion/checkpoints/clip"
                
                if "cond_stage_config" in config.model.params:
                    if "params" in config.model.params.cond_stage_config:
                        config.model.params.cond_stage_config.params.cache_dir = "/tmp/stable_diffusion/checkpoints/clip"

                # Update data paths
                if "data" in config and "params" in config.data:
                    if "train" in config.data.params and "params" in config.data.params.train:
                        if "urls" in config.data.params.train.params:
                            config.data.params.train.params.urls = "/tmp/stable_diffusion/datasets/laion-400m/webdataset-moments-filtered/{00000..00831}.tar"
                    if "validation" in config.data.params and "params" in config.data.params.validation:
                        if "annotations_file" in config.data.params.validation.params:
                            config.data.params.validation.params.annotations_file = "/tmp/stable_diffusion/datasets/coco2014/val2014_30k.tsv"

                # Create all necessary directories
                os.makedirs("/tmp/stable_diffusion/checkpoints/inception", exist_ok=True)
                os.makedirs("/tmp/stable_diffusion/checkpoints/clip", exist_ok=True)
                os.makedirs("/tmp/stable_diffusion/checkpoints/models", exist_ok=True)
                os.makedirs("/tmp/stable_diffusion/results", exist_ok=True)
                os.makedirs("/tmp/stable_diffusion/datasets/coco2014", exist_ok=True)
                os.makedirs("/tmp/stable_diffusion/datasets/laion-400m/webdataset-moments-filtered", exist_ok=True)

                # Set environment variables for model caching
                os.environ["TORCH_HOME"] = "/tmp/stable_diffusion/checkpoints"  # For torch hub cache
                os.environ["HF_HOME"] = "/tmp/stable_diffusion/checkpoints/huggingface"  # For HuggingFace cache
                os.environ["XDG_CACHE_HOME"] = "/tmp/stable_diffusion/checkpoints"  # For general cache
            else:
                raise ValueError(f"Default config file not found at {default_config_path}")
        else:
            configs = [OmegaConf.load(cfg) for cfg in opt.base]
            cli = OmegaConf.from_dotlist(unknown)
            config = OmegaConf.merge(*configs, cli)

        # Extract lightning config
        lightning_config = config.pop("lightning", OmegaConf.create())

        # merge trainer cli with config
        trainer_config = lightning_config.get("trainer", OmegaConf.create())

        # Update trainer configuration to use 1 GPU
        if "lightning" in config:
            if "trainer" in config.lightning:
                config.lightning.trainer.devices = 1
                config.lightning.trainer.num_nodes = 1
                if "strategy" in config.lightning.trainer:
                    del config.lightning.trainer.strategy

        # Set defaults for trainer config
        default_trainer_config = {
            "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
            "devices": 1,
            "num_nodes": 1,
            "precision": "32-true",
            "logger": True,
            "callbacks": [],
            "max_epochs": 1000,
        }
        for k, v in default_trainer_config.items():
            if k not in trainer_config:
                trainer_config[k] = v

        if trainer_config.get("accelerator") != "gpu":
            cpu = True
            rank_zero_info("Using CPU")
        else:
            cpu = False
            if trainer_config.get("devices") > 1:
                rank_zero_info(f"Using {trainer_config.devices} GPUs")
            else:
                rank_zero_info("Using single GPU")

        # Determine device based on trainer config
        device = "cuda" if torch.cuda.is_available() and trainer_config.get("accelerator", "gpu") == "gpu" else "cpu"
        
        # Create model
        model = instantiate_from_config(config.model)
        model = model.to(device)
        
        # Ensure all submodules are on the same device
        for module in model.modules():
            # Only set device attribute if it exists and is writable
            if hasattr(module, 'device') and isinstance(getattr(type(module), 'device', None), property):
                if hasattr(type(module).device, 'fset') and type(module).device.fset is not None:
                    module.device = device
            module.to(device)

        # Update trainer config to ensure data is moved to correct device
        if device == "cuda":
            trainer_config.accelerator = "gpu"
            trainer_config.devices = 1
        else:
            trainer_config.accelerator = "cpu"
            trainer_config.devices = None

        # Set defaults for trainer config
        default_trainer_config = {
            "accelerator": "gpu" if device == "cuda" else "cpu",
            "devices": 1 if device == "cuda" else None,
            "num_nodes": 1,
            "precision": "32-true",
            "logger": True,
            "callbacks": [],
            "max_epochs": 1000,
        }

        for k in default_trainer_config:
            if k not in trainer_config:
                trainer_config[k] = default_trainer_config[k]

        trainer_kwargs = OmegaConf.to_container(trainer_config)

        # Add model-specific callbacks
        # Callbacks are always called with model and device
        callbacks = [SetupCallback(resume=opt.resume, now=now, logdir=logdir, ckptdir=ckptdir,
                                 cfgdir=cfgdir, config=config, lightning_config=lightning_config)]

        if "callbacks" in lightning_config:
            for key in lightning_config.callbacks:
                logger.info(f"Adding callback <{key}>")
                callbacks.append(instantiate_from_config(lightning_config.callbacks[key]))

        trainer_kwargs["callbacks"] = callbacks

        # config the logger
        # Default logger configs to  log training metrics during the training process.
        # These loggers are specified as targets in the dictionary, along with the configuration settings specific to each logger.
        default_logger_cfgs = {
            "wandb": {
                "target": LIGHTNING_PACK_NAME + "loggers.WandbLogger",
                "params": {
                    "name": nowname,
                    "save_dir": logdir,
                    "offline": opt.debug,
                    "id": nowname,
                }
            },
            "tensorboard": {
                "target": LIGHTNING_PACK_NAME + "loggers.TensorBoardLogger",
                "params": {
                    "save_dir": logdir,
                    "name": "diff_tb",
                    "log_graph": True
                }
            }
        }

        # Set up the logger for TensorBoard
        default_logger_cfg = default_logger_cfgs["tensorboard"]
        if "logger" in lightning_config:
            logger_cfg = lightning_config.logger
        else:
            logger_cfg = default_logger_cfg
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

        # config the strategy, defualt is ddp
        if "strategy" in trainer_config:
            strategy_cfg = trainer_config["strategy"]
            strategy_cfg["target"] = LIGHTNING_PACK_NAME + strategy_cfg["target"]
        else:
            strategy_cfg = {
                "target": LIGHTNING_PACK_NAME + "strategies.DDPStrategy",
                "params": {
                    "find_unused_parameters": False
                }
            }

        trainer_kwargs["strategy"] = instantiate_from_config(strategy_cfg)

        # Set up various callbacks, including logging, learning rate monitoring, and CUDA management
        # add callback which sets up log directory
        default_callbacks_cfg = {
            "setup_callback": {                           # callback to set up the training
                "target": "main.SetupCallback",
                "params": {
                    "resume": opt.resume,                 # resume training if applicable
                    "now": now,
                    "logdir": logdir,                     # directory to save the log file
                    "ckptdir": ckptdir,                   # directory to save the checkpoint file
                    "cfgdir": cfgdir,                     # directory to save the configuration file
                    "config": config,                     # configuration dictionary
                    "lightning_config": lightning_config, # LightningModule configuration
                }
            },
            "learning_rate_logger": {                     # callback to log learning rate
                "target": "lightning.pytorch.callbacks.LearningRateMonitor",
                "params": {
                    "logging_interval": "step",           # logging frequency (either 'step' or 'epoch')
                }
            },
            "cuda_callback": {                            # callback to handle CUDA-related operations
                "target": "main.CUDACallback"
            },
            "fid_clip_early_stop_callback" : {
                "target": "main.ListEarlyStopping",
                "params": {
                    "stopping_thresholds": {"validation/fid": opt.fid_threshold, "validation/clip": opt.clip_threshold},
                    "check_finite": True
                }
            },
        }

        # If the LightningModule configuration has specified callbacks, use those
        # Otherwise, create an empty OmegaConf configuration object
        if "callbacks" in lightning_config:
            callbacks_cfg = lightning_config.callbacks
        else:
            callbacks_cfg = OmegaConf.create()


        # Merge the default callbacks configuration with the specified callbacks configuration, and instantiate the callbacks
        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)

        trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
        trainer_kwargs["callbacks"].append(mlperf_logging_utils.MLPerfLoggingCallback(logger=mllogger,
                                                                                      train_log_interval=opt.train_log_interval,
                                                                                      validation_log_interval=opt.validation_log_interval))

        # Set up ModelCheckpoint callback to save best models
        # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
        # specify which metric is used to determine best models
        default_modelckpt_cfg = {
            "target": "lightning.pytorch.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": "/tmp/stable_diffusion/checkpoints/models",
                "filename": "{epoch:06}",
                "verbose": True,
                "save_last": True,
                "every_n_train_steps": 1000,
                "save_top_k": -1,
            }
        }

        if "modelcheckpoint" in lightning_config:
            modelckpt_cfg = lightning_config.modelcheckpoint
        else:
            modelckpt_cfg = OmegaConf.create()
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        trainer_kwargs["callbacks"].append(instantiate_from_config(modelckpt_cfg))

        # Check if real datasets are available
        laion_path = "/tmp/stable_diffusion/datasets/laion-400m/webdataset-moments-filtered/00000.tar"
        coco_path = "/tmp/stable_diffusion/datasets/coco2014/val2014_30k.tsv"
        
        if not os.path.exists(laion_path) or not os.path.exists(coco_path):
            rank_zero_info("Required datasets not found. Using dummy data.")
            rank_zero_info(f"LAION dataset exists: {os.path.exists(laion_path)}")
            rank_zero_info(f"COCO dataset exists: {os.path.exists(coco_path)}")
            
            # Store original batch sizes
            train_batch_size = config.data.params.train.params.batch_size if "data" in config and "params" in config.data and "train" in config.data.params and "params" in config.data.params.train and "batch_size" in config.data.params.train.params else 8
            val_batch_size = config.data.params.validation.params.batch_size if "data" in config and "params" in config.data and "validation" in config.data.params and "params" in config.data.params.validation and "batch_size" in config.data.params.validation.params else 8
            
            # Switch to dummy data configuration
            config.data = {
                "target": "dummy",
                "params": {
                    "train": {
                        "params": {
                            "batch_size": train_batch_size
                        }
                    },
                    "validation": {
                        "params": {
                            "batch_size": val_batch_size
                        }
                    },
                    "image_size": config.model.params.image_size,
                    "num_samples": 1000
                }
            }

        # Create a Trainer object with the specified command-line arguments and keyword arguments, and set the log directory
        trainer = Trainer.from_argparse_args(opt, **trainer_kwargs)
        trainer.logdir = logdir

        # Create a data module based on the configuration file
        if hasattr(config.data, "target") and config.data.target == "dummy":
            data = DummyDataModule(
                batch_size=config.data.params.train.params.batch_size,
                image_size=config.data.params.image_size,
                num_samples=config.data.params.num_samples
            )
        else:
            data = instantiate_from_config(config.data)
        # We can't get number of samples without reading the data (which we can't inside the init block), so we hard code them
        mllogger.event(key=mllog_constants.TRAIN_SAMPLES, value=6513144)
        mllogger.event(key=mllog_constants.EVAL_SAMPLES, value=30000)

        # Configure gradient accumulation
        if 'accumulate_grad_batches' in lightning_config.trainer:
            accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
        else:
            accumulate_grad_batches = 1
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        mllogger.event(mllog_constants.GRADIENT_ACCUMULATION_STEPS, value=accumulate_grad_batches)

        # Configure number of GPUs
        if not cpu:
            ngpu = trainer_config["devices"] * trainer_config["num_nodes"]
        else:
            ngpu = 1

        # Configure batch size
        local_batch_size = config.data.params.train.params.batch_size
        global_batch_size = local_batch_size*ngpu
        mllogger.event(mllog_constants.GLOBAL_BATCH_SIZE, value=trainer.world_size * local_batch_size)

        # Configure learning rate based on the batch size, base learning rate and number of GPUs
        # If scale_lr is true, calculate the learning rate based on additional factors
        base_lr = config.model.base_learning_rate
        if opt.scale_lr:
            model.learning_rate = accumulate_grad_batches * ngpu * local_batch_size * base_lr
            rank_zero_info(
                "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (local_batch_size) * {:.2e} (base_lr)"
                .format(model.learning_rate, accumulate_grad_batches, ngpu, local_batch_size, base_lr))
        else:
            model.learning_rate = base_lr
            rank_zero_info("++++ NOT USING LR SCALING ++++")
            rank_zero_info(f"Setting learning rate to {model.learning_rate:.2e}")

        # Allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                ckpt_path = os.path.join("/tmp/stable_diffusion/checkpoints/models", "last.ckpt")
                trainer.save_checkpoint(ckpt_path)

        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb
                pudb.set_trace()

        import signal
        # Assign melk to SIGUSR1 signal and divein to SIGUSR2 signal
        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)

        mllogger.end(mllog_constants.INIT_STOP)

        # Run the training and validation
        if opt.mode=="train":
            try:
                trainer.fit(model, data)
            except Exception:
                melk()
                raise
        elif opt.mode=="validate":
            trainer.validate(model, data)
        else:
            raise ValueError(f"Unknown mode {opt.mode}")

        # Default is True in case thresholds are not defined
        fid_success = True
        clip_success = True
        if opt.fid_threshold is not None:
            fid_success =  "validation/fid" in trainer.callback_metrics and opt.fid_threshold >= trainer.callback_metrics["validation/fid"].item()
        if opt.clip_threshold is not None:
            clip_success = "validation/clip" in trainer.callback_metrics and opt.clip_threshold <= trainer.callback_metrics["validation/clip"].item()

        status = mllog_constants.SUCCESS if fid_success and clip_success else mllog_constants.ABORTED

    except Exception:
        # If there's an exception, debug it if opt.debug is true and the trainer's global rank is 0
        if opt.debug and trainer.global_rank == 0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        #  Move the log directory to debug_runs if opt.debug is true and the trainer's global
        if opt.debug and not opt.resume and trainer is not None and trainer.global_rank == 0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)
        if trainer is not None:  # Only log status if trainer was created
            mllogger.event(mllog_constants.STATUS, value=status)
