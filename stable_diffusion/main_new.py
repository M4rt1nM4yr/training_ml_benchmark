import os
import torch
import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

class DummyDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=4, image_size=64, num_samples=100):
        super().__init__()
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_samples = num_samples
        self.device = 'cpu'

    def setup(self, stage=None):
        self.random_data = torch.randn(self.num_samples, 3, self.image_size, self.image_size)
        self.random_captions = ["dummy caption"] * self.num_samples

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            list(zip(self.random_data, self.random_captions)),
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )

    def val_dataloader(self):
        return self.train_dataloader()

    def _collate_fn(self, batch):
        images = torch.stack([item[0] for item in batch])
        captions = [item[1] for item in batch]
        return {"jpg": images, "txt": captions}

def main():
    # Set random seed for reproducibility
    seed_everything(42)

    # Load config
    config = OmegaConf.load("stable_diffusion/configs/train_01x08x08_noCkpt.yaml")
    
    # Create model
    model = instantiate_from_config(config.model)
    
    # Create dummy data module
    data = DummyDataModule(
        batch_size=4,
        image_size=64,
        num_samples=100
    )

    # Create trainer
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=1,
        precision="32",
    )

    # Train model
    trainer.fit(model, data)

if __name__ == "__main__":
    main()