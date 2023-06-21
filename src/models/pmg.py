'''
Progressive Multi-Granularity Training, from "Fine-Grained Visual Classification
via Progressive Multi-Granularity Training of Jigsaw Patches" 
(https://arxiv.org/abs/2003.03836).

Adapted from https://github.com/PRIS-CV/PMG-Progressive-Multi-Granularity-Training
'''
from typing import Optional

import torch
import torchmetrics

from .base import ImageClassifier


__all__ = [
    'PMG',
]


################################################################################
# Internal Modules
################################################################################

class Classifier(torch.nn.Module):
    def __init__(self, fin, fmid, fout):
        super().__init__()
        self.bn1 = torch.nn.BatchNorm1d(fin)
        self.fc1 = torch.nn.Linear(fin ,fmid)
        self.bn2 = torch.nn.BatchNorm1d(fmid)
        self.fc2 = torch.nn.Linear(fmid, fout)
        self.act = torch.nn.functional.elu_
    
    def forward(self, x):
        x = x.flatten(1)
        x = self.fc1(self.bn1(x))
        x = self.fc2(self.act(self.bn2(x)))
        return x


class ConvBlock(torch.nn.Module):
    def __init__(self, fin, fmid, fout):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(fin, fmid, 1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(fmid)
        self.conv2 = torch.nn.Conv2d(fmid, fout, 3, 1, 1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(fout)
        self.act = torch.nn.functional.relu_

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return x


################################################################################
# Lightning Module for PMG
################################################################################

class PMG(ImageClassifier):
    '''Progressive Multi-Granularity Training of Jigsaw Patches (PMG).

    Args:
        feature_size (int): 
    '''
    def __init__(
        self,
        feature_size: int=512,
        base_conf: Optional[dict]=None,
        model_conf: Optional[dict]=None,
    ):
        self.feature_size = feature_size

        # parent class initialization
        ImageClassifier.__init__(self, base_conf=base_conf, model_conf=model_conf)

        # enable manual optimization, since PMG performs multiple forward/backward passes per batch
        self.automatic_optimization = False

    def inject_backbone_args(self):
        # PMG uses features extracted from multiple stages of the network
        self.model_conf.model_kw.update(
            features_only=True,
            out_indices=(2, 3, 4),  # last 3 stages of the ResNet backbone
        )
    
    def setup_model(self):
        with torch.no_grad():
            _x = torch.ones(1, 3, 33, 33)
            channels = [v.shape[1] for v in self.backbone(_x)]

        self.maxpool = torch.nn.AdaptiveMaxPool2d((1, 1))
        
        # conv blocks
        self.conv_blocks = torch.nn.ModuleList([
            ConvBlock(c, self.feature_size, channels[-1] // 2)
            for c in channels
        ])

        # classifier blocks
        self.classifiers = torch.nn.ModuleList([
            Classifier(channels[-1] // 2, self.feature_size, self.num_classes)
            for _ in range(3)
        ])
        self.classifier_concat = Classifier(3 * (channels[-1] // 2), self.feature_size, self.num_classes)

    def setup_metrics(self):
        super().setup_metrics()
        self.val_acc_combined = torchmetrics.Accuracy('multiclass', num_classes=self.num_classes)

    def forward(self, x, level=None):
        # level can be 0, 1, 2, 3, or None (meaning all), specifying 
        # which of the intermediate outputs to compute and return
        layer_outs = self.backbone(x)

        # first 3 levels correspond to one of the granularity classifiers
        if level in (0, 1, 2):
            x = self.conv_blocks[level](layer_outs[level])
            x = self.maxpool(x)
            x = self.classifiers[level](x)
            return x

        xs = []  # conv block outputs
        for xi, conv_block in zip(layer_outs, self.conv_blocks):
            xs.append(self.maxpool(conv_block(xi)))

        ycat = self.classifier_concat(torch.cat(xs, -1))

        # 4th level is the concatenated classifier
        if level == 3:
            return ycat
        
        # if levels is None, then return all the classifier outputs
        ys = []  # classifier outputs
        for xi, classifier in zip(xs, self.classifiers):
            ys.append(classifier(xi.view(xi.shape[0], -1)))
        ys.append(ycat)

        return ys

    @staticmethod
    def jigsaw_generator(images: torch.Tensor, n: int):
        b, c, h, w = images.shape
        hn, wn = h//n, w//n
        s1 = (b, c, n, hn, n, wn)
        s2 = (b, n, n, c, hn, wn)
        p1 = (0, 2, 4, 1, 3, 5)
        p2 = (0, 3, 1, 4, 2, 5)
        # s2 = [b, c, n**2, hn, wn]
        # s3 = [b, c, n, n, hn, wn]
        # p = [0, 1, 2, 4, 3, 5]

        # idx = torch.multinomial(torch.ones(n**2, device=images.device), n**2)
        # jigsaw = images.view(s1).permute(p).reshape(s2)[:,:,idx]
        # jigsaw = jigsaw.reshape(s3).permute(p).reshape(images.shape)

        bi = torch.arange(b, device=images.device)[:, None]
        idx = torch.multinomial(torch.ones(b, n**2, device=images.device), n**2)
        jigsaw = images.view(s1).permute(p1).flatten(1, 2)[bi, idx]
        jigsaw = jigsaw.reshape(s2).permute(p2).reshape(images.shape)

        return jigsaw.contiguous()

    def training_step(self, batch, batch_idx):
        x, y = batch
        opt = self.optimizers()
        losses = []

        # forward and backward pass for each of the granularities
        ns = [8, 4, 2, 1]
        for i, n in enumerate(ns):
            n = ns[i]
            if n > 1:
                js = self.jigsaw_generator(x, n)
            else:
                js = x
            v = self(js, level=i)
            loss = self.objective(v, y)
            if n == 1:
                loss = loss * 2
            opt.zero_grad()
            self.manual_backward(loss)
            opt.step()
            losses.append(loss.detach())

        # combined metrics
        acc = self.train_accuracy(v, y)
        total_loss = sum(losses)
        losses = {f'train/loss_{i}': ll for i, ll in enumerate(losses)}

        self.log('train/loss', total_loss, prog_bar=True)
        self.log('train/acc', acc, prog_bar=True)
        self.log_dict(losses)

        return total_loss.detach()
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        outs = self(x, level=None)
        loss = self.objective(outs[-1], y)
        acc = self.val_accuracy(outs[-1], y)

        logits = sum(outs)
        acc_comb = self.val_acc_combined(logits, y)

        log_kw = dict(on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/loss', loss, prog_bar=True, **log_kw)
        self.log('val/acc', acc, prog_bar=True, **log_kw)
        self.log('val/acc_comb', acc_comb, prog_bar=True, **log_kw)
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        outs = self(x, level=None)
        logits = sum(outs)
        return {'logits': logits, 'y': y}
    