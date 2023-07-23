from typing import Any, Dict, Optional

import hydra
from omegaconf import DictConfig
from lightning import LightningDataModule
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

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        if stage == "fit":
            self.train_dataset = hydra.utils.instantiate(
                self.hparams.dataset,
                ann_filename=self.hparams.train_ann_filename,
            )
            self.val_dataset = hydra.utils.instantiate(
                self.hparams.dataset,
                ann_filename=self.hparams.val_ann_filename,
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        self.train_dataset = None
        self.val_dataset = None

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass
