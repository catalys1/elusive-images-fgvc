'''
'''
import fgvcdata
import numpy as np
import torch
import torchvision.transforms as T

from . base import BaseDataModule


def ToTensor(x):
    return torch.from_numpy(np.array(x)).permute(2,0,1)


class FGVCDataModule(BaseDataModule):
    def __init__(self, normalize: str='in1k', **kwargs):
        super().__init__(**kwargs)
        self.normalize = normalize

    def transforms(self):
        if self.normalize == 'in1k':
            # normalize with imagenet-1k statistics
            nrm = [
                T.ToTensor(),
                T.Normalize(fgvcdata.IMAGENET_STATS)
            ]
        elif self.normalize == 'in21k':
            # normalize to range [0, 1]
            nrm = [T.ToTensor()]
        elif self.normalize == 'none':
            # convert to tensor without normalization: uint8 [0, 255]
            nrm = [ToTensor()]

        train = T.Compose([
            T.RandomResizedCrop((self.size, self.size)),
            T.RandomHorizontalFlip(0.5),
            T.ColorJitter(.1,.1,.1),
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


class CUB(FGVCDataModule):
    dataclass = fgvcdata.CUB
    num_classes = 200