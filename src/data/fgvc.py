'''
'''
from typing import Any
import fgvcdata
import numpy as np
import torch
import torchvision.transforms as T

from . base import BaseDataModule


__all__ = [
    'CUB',
]


def ToTensor(x):
    return torch.from_numpy(np.array(x)).permute(2,0,1)


class FGVCDataModule(BaseDataModule):
    def __init__(self, normalize: str='in1k', normalize_on_gpu: bool=False, **kwargs):
        super().__init__(**kwargs)

        supported_norm = ('in1k', 'in21k')
        if normalize not in supported_norm:
            raise RuntimeError(
                f'"{normalize}" is not a supported normalization type. '
                f'Available options are {str(supported_norm)}'
            )
        self.normalize = normalize
        self.normalize_on_gpu = normalize_on_gpu

    def transforms(self):
        if self.normalize_on_gpu:
            # convert to tensor without normalization: uint8 [0, 255]
            nrm = [ToTensor]
        else:
            if self.normalize == 'in1k':
                # normalize with imagenet-1k statistics
                nrm = [
                    T.ToTensor(),
                    T.Normalize(*fgvcdata.IMAGENET_STATS)
                ]
            elif self.normalize == 'in21k':
                # normalize to range [0, 1]
                nrm = [T.ToTensor()]

        train = T.Compose([
            T.RandomResizedCrop((self.size, self.size), (0.1, 1), (0.8, 1.25)),
            T.ColorJitter(0.25, 0.25, 0.25),
            T.RandomHorizontalFlip(0.5),
            *nrm,
        ])
        
        resize = [round(8/7*self.size/32)*32] * 2
        val = T.Compose([
            T.Resize(resize),
            T.CenterCrop((self.size, self.size)),
            *nrm,
        ])

        return train, val

    def setup(self, stage: str='fit'):
        tf_train, tf_val = self.transforms()

        if stage == 'fit':
            self.train_data = self.dataclass(self.root / 'train', tf_train)

        self.val_data = self.dataclass(self.root / 'val', tf_val)

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        if self.normalize_on_gpu:
            if self.normalize == 'in1k':
                batch[0] = T.functional.normalize(batch[0].float().div_(255), *fgvcdata.IMAGENET_STATS)
            elif self.normalize == 'in21k':
                batch[0] = batch[0].float().div_(255)
        return batch


class CUB(FGVCDataModule):
    dataclass = fgvcdata.CUB
    num_classes = 200
