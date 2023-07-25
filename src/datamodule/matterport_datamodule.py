from typing import Any, Dict, Optional

import hydra
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class MatterportDatamodule(LightningDataModule):
    def __init__(
            self,
            train_ann_filename: str,
            val_ann_filename: str,
            dataset: DictConfig,
            batch_size: int,
            num_workers: int = 0,
            pin_memory: bool = False,
    ):
        super().__init__()
        self.train_ann_filename = train_ann_filename
        self.val_ann_filename = val_ann_filename
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        if stage == "fit":
            self.train_dataset = hydra.utils.instantiate(
                self.dataset,
                ann_filename=self.train_ann_filename,
            )
            self.val_dataset = hydra.utils.instantiate(
                self.dataset,
                ann_filename=self.val_ann_filename,
            )
        else:
            raise f"[{stage}] is not support yet!"

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        self.train_dataset = None
        self.val_dataset = None
