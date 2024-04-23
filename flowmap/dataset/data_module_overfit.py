from typing import Iterator

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, IterableDataset


class DummyDataset(IterableDataset):
    def __init__(self, limit: int | None = None) -> None:
        self.limit = limit

    def __iter__(self) -> Iterator:
        if self.limit:
            for _ in range(self.limit):
                yield []
        else:
            while True:
                yield []


class DataModuleOverfit(LightningDataModule):
    """Give PyTorch Lightning dummy data loaders so it runs the training loop. The
    ModelWrapper already has the training data from its __init__.
    """

    def train_dataloader(self):
        return DataLoader(DummyDataset(), 1, num_workers=0)

    def val_dataloader(self):
        return DataLoader(DummyDataset(limit=1), 1, num_workers=0)
