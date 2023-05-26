'''Base LightningDataModule to be inherited for each dataset.
'''
from pathlib import Path

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, default_collate


class BaseDataModule(LightningDataModule):
    '''BaseDataModule.

    Args:
        root (str): path to the dataset root folder.
        batch_size (int): batch size.
        num_workers (int): number of worker processes for dataloading (default: 0).
        shuffle (bool): whether to shuffle the training set each epoch (default: True).
        pin_memory (bool): whether to pin GPU memory (default: True).
        drop_last (bool): whether to drop the last training batch if its a partial (default: True)
    '''
    def __init__(
        self,
        root: str,
        batch_size: int,
        num_workers: int=0,
        shuffle: bool=True,
        pin_memory: bool=True,
        drop_last: bool=True,
        size: int=224,
    ):
        super().__init__()
        self.root = Path(root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.size = size

        self.post_init()

    def post_init(self):
        pass

    def transforms(self):
        raise NotImplementedError()

    def setup(self, stage: str='fit'):
        raise NotImplementedError()

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            collate_fn=self.collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            drop_last=False,
            collate_fn=self.collate,
        )

    def collate(self, batch):
        # Override for custom collate function
        return default_collate(batch)