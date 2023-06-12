'''
'''
from typing import Any
import fgvcdata
import numpy as np
import torch
import torchvision.transforms as T

from . base import BaseDataModule


__all__ = [
    'Aircraft',
    'CUB',
    'NABirds',
    'StanfordCars',
]


def ToTensor(x):
    return torch.from_numpy(np.array(x)).permute(2,0,1)


class FGVCDataModule(BaseDataModule):
    def __init__(
        self,
        normalize: str='in21k',
        normalize_on_gpu: bool=True,
        multi_augment: int=0,
        **kwargs
    ):
        super().__init__(**kwargs)

        supported_norm = ('in1k', 'in21k')
        if normalize not in supported_norm:
            raise RuntimeError(
                f'"{normalize}" is not a supported normalization type. '
                f'Available options are {str(supported_norm)}'
            )
        self.normalize = normalize
        self.normalize_on_gpu = normalize_on_gpu
        self.multi_augment = multi_augment

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

        if self.multi_augment > 0:
            train = T.Compose([
                T.RandomResizedCrop((self.size, self.size), (0.25, 1), (0.9, 1 / 0.9)),
                *nrm,
            ])
        else:
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

        if self.multi_augment > 0:
            self.gpu_tform = torch.nn.Sequential(
                T.RandomResizedCrop(self.size, (0.1, 1), (0.8, 1.25), antialias=True),
                T.ColorJitter(0.25, 0.25, 0.25),
                T.RandomHorizontalFlip(0.5),
            )

        if stage == 'fit':
            self.train_data = self.dataclass(self.root / 'train', tf_train)

        self.val_data = self.dataclass(self.root / 'val', tf_val)

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        if self.trainer.training and self.multi_augment > 0:
            batch[0] = torch.cat([self.gpu_tform(batch[0]) for _ in range(self.multi_augment)])
            batch[1] = batch[1].repeat(self.multi_augment)

        if self.normalize_on_gpu:
            if self.normalize == 'in1k':
                batch[0] = T.functional.normalize(batch[0].float().div_(255), *fgvcdata.IMAGENET_STATS)
            elif self.normalize == 'in21k':
                batch[0] = batch[0].float().div_(255)
        return batch



class Aircraft(FGVCDataModule):
    dataclass = fgvcdata.Aircraft
    num_classes = 100


class CUB(FGVCDataModule):
    dataclass = fgvcdata.CUB
    num_classes = 200


class NABirds(FGVCDataModule):
    dataclass = fgvcdata.NABirds
    num_classes = 556


class StanfordCars(FGVCDataModule):
    dataclass = fgvcdata.StanfordCars
    num_classes = 196