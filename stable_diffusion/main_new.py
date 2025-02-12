import os
import torch
import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

class DummyDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=4, image_size=64, num_samples=100, use_fp16=True, num_workers=12):
        super().__init__()
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_samples = num_samples
        self.device = 'cpu'
        self.use_fp16 = use_fp16
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Set precision based on config
        dtype = torch.float16 if self.use_fp16 else torch.float32
        self.random_data = torch.randn(
            self.num_samples, 3, self.image_size, self.image_size, 
            dtype=dtype,
            device=self.device
        )
        self.random_captions = ["dummy caption"] * self.num_samples

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            list(zip(self.random_data, self.random_captions)),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            list(zip(self.random_data, self.random_captions)),
            batch_size=self.batch_size,
            shuffle=False,  # No shuffling for validation
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn
        )

    def _collate_fn(self, batch):
        images = torch.stack([item[0] for item in batch])
        captions = [item[1] for item in batch]
        # Return both the original keys and the keys needed for validation
        return {
            "jpg": images,
            "txt": captions,
            "caption": captions,  # Add this key for validation
            "image_id": [f"dummy_{i}.jpg" for i in range(len(batch))]  # Use image_id as specified in config
        }

def main():
    # Set random seed for reproducibility
    seed_everything(42)

    # Load config
    config = OmegaConf.load("stable_diffusion/configs/train_01x08x08_noCkpt.yaml")
    
    # Create model
    model = instantiate_from_config(config.model)
    
    model.learning_rate = 1e-4

    # Create dummy data module
    data_module = DummyDataModule(
        batch_size=2,
        image_size=512,
        num_samples=100,
        use_fp16=config.model.params.unet_config.params.use_fp16,
        num_workers=12  # Use number of CPU cores
    )

    # Set precision based on config
    precision = "16" if config.model.params.unet_config.params.use_fp16 else "32"
    
    # Initialize trainer
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=1,
        precision=precision,
    )

    # Train model
    trainer.fit(model, data_module)

if __name__ == "__main__":
    main()