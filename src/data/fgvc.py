
import fgvcdata
import torchvision.transforms as T

from . base import BaseDataModule


class FGVCDataModule(BaseDataModule):
    def transforms(self):
        train = T.Compose([
            T.RandomResizedCrop((self.size, self.size)),
            T.RandomHorizontalFlip(0.5),
            T.ColorJitter(.1,.1,.1),
            T.ToTensor(),
            T.Normalize(*fgvcdata.IMAGENET_STATS),
        ])
        
        resize = [round(8/7*self.size/32)*32] * 2
        val = T.Compose([
            T.Resize(resize),
            T.CenterCrop((self.size, self.size)),
            T.ToTensor(),
            T.Normalize(*fgvcdata.IMAGENET_STATS),
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