import os
from argparse import ArgumentParser
import torch
from typing import Optional
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import random_split
import torchvision.transforms as T
from torchvision.datasets import MNIST


class MNISTDatamodule(pl.LightningDataModule):
    def __init__(self,
                 root=os.path.expanduser("~/.cache"),
                 no_pad=False,
                 batch_size=256,
                 num_workers=10,
                 train_val_split=0.8,
                 seed=None,
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.transforms = None
        self.train_ds, self.val_ds, self.test_ds = None, None, None

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser, **kwargs) -> ArgumentParser:
        parser = parent_parser.add_argument_group("Data")
        parser.add_argument("--root", type=str, required=False, default=os.path.expanduser("~/.cache"), help="path to folder where data will be stored")
        parser.add_argument("--no_pad", default=False, action="store_true", help="set in order to keep image 28x28 (otherwise images will padded to 32x32)")
        parser.add_argument("--batch_size", type=int, default=256, help="training batch size")
        parser.add_argument("--num_workers", type=int, default=10, help="number of CPUs available")
        parser.add_argument("--train_val_split", type=float, default=0.8, help="train-validation split coefficient")
        return parent_parser

    def _dataloader(self, mode):
        return DataLoader(
            getattr(self, f'{mode}_ds'),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,  # must pin memory for DDP
            shuffle=True if mode == 'train' else False
        )

    def train_dataloader(self):
        return self._dataloader('train')

    def val_dataloader(self):
        return self._dataloader('val')

    def test_dataloader(self):
        return self._dataloader('test')

    def prepare_data(self) -> None:
        MNIST(self.hparams.root, download=True, train=True)
        MNIST(self.hparams.root, download=True, train=False)

    def setup(self, stage: Optional[str] = None) -> None:
        transforms = T.ToTensor() if self.hparams.no_pad else T.Compose([T.Pad(2), T.ToTensor()])
        self.train_ds = MNIST(self.hparams.root, download=False, train=True, transform=transforms)
        self.test_ds = MNIST(self.hparams.root, download=False, train=False, transform=transforms)
        if self.hparams.train_val_split < 1:
            train_size = int(len(self.train_ds) * self.hparams.train_val_split)
            seed_generator = torch.Generator().manual_seed(self.hparams.seed) if self.hparams.seed is not None else None
            split = [train_size, len(self.train_ds) - train_size]
            self.train_ds, self.val_ds = random_split(self.train_ds, split, seed_generator)
