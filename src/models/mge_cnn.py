'''
MGE-CNN, from "Learning Granularity-Aware Convolutional Neural Network for Fine-Grained Visual Classification" 
(https://arxiv.org/abs/2103.02788).

Adapted from https://github.com/lbzhang/MGE-CNN
'''
from typing import Optional

import torch

from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

from .base import ImageClassifier, get_backbone

__all__ = [
    'MGE-CNN',
]

class MGE(ImageClassifier):
    '''
    Lightning Module for MGE-CNN
    '''
    def __init__(
        self,
        base_conf: Optional[dict]=None,
        model_conf: Optional[dict]=None,
        conv4, conv5, pool
    ):
        # parent class initialization
        ImageClassifier.__init__(self, base_conf=base_conf, model_conf=model_conf)

        # Main branch
        basenet = self.model_conf

        self.conv4 = nn.Sequential(*list(basenet.children())[:-3])
        self.conv5 = nn.Sequential(*list(basenet.children())[-3])

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = Classifier(2048, self.num_classes, bias-True)

        # Other branch
        self.conv4_box = copy.deepcopy(self.conv4)
        self.conv5_box = copy.deepcopy(self.conv5)
        self.classifier_box = Classifier(2048, self.num_classes, bias=True)

        self.conv4_box_2 = copy.deepcopy(self.conv4)
        self.conv5_box_2 = copy.deepcopy(self.conv5)

        self.classifier_box_2 = Classifier(2048, self.num_classes, bias=True)

        # part information
        self.conv6_1 = nn.Conv2d(1024, 10*self.num_classes, 1, 1, 1)
        self.conv6_2 = nn.Conv2d(1024, 10*self.num_classes, 1, 1, 1)
        self.conv6 = nn.Conv2d(1024, 10*self.num_classes, 1, 1, 1)

        self.cls_part_1 = Classifier(10*self.num_classes, self.num_classes, bias=True)
        self.cls_part_2 = Classifier(10*self.num_classes, self.num_classes, bias=True)
        self.cls_part = Classifier(10*self.num_classes, self.num_classes, bias=True)

        self.cls_cat_1 = Classifier(2048+10*self.num_classes, self.num_classes, bias=True)
        self.cls_cat_2 = Classifier(2048+10*self.num_classes, self.num_classes, bias=True)
        self.cls_cat = Classifier(2048+10*self.num_classes, self.num_classes, bias=True)

        self.cls_cat_a = Classifier(3*(2048+10*self.num_classes), self.num_classes, bias=True)
        self.self.num_classes = self.num_classes
        self.box_thred = opt.box_thred

        # gating network
        self.conv4_gate = copy.deepcopy(self.conv4)
        self.conv5_gate = copy.deepcopy(self.conv5)
        self.cls_gate = nn.Sequential(Classifier(2048, 512, bias=True), Classifier(512, 3, bias=True))

    
    def forward(self, x):
        pass
    
    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass
    

class Simple(ImageClassifier):
    '''
    Trivial network for testing purposes
    '''
    def __init__(
        self,
        base_conf: Optional[dict]=None,
        model_conf: Optional[dict]=None,
    ):
        # parent class initialization
        ImageClassifier.__init__(self, base_conf=base_conf, model_conf=model_conf)

        # simple CNN model with one convolutional layer and one fully connected layer
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        #self.fc = nn.Linear(16 * 28 * 28, 10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = torch.flatten(x, 1)
        # x = self.fc(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer