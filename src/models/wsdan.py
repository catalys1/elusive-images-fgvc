'''
Weakly Supervised Data Augmentation Network, from "See Better Before Looking
Closer: Weakly Supervised Data Augmentation Network for Fine-Grained Visual
Classification" (https://arxiv.org/abs/1901.09891).

Adapted from https://github.com/GuYuc/WS-DAN.PyTorch
'''
from functools import partial
import random
from typing import Optional

import torch
from torchmetrics import Accuracy

from .base import ImageClassifier


__all__ = ['WSDAN']


# SETTINGS FROM THE REPOSITORY
# lr: 1e-3
# optimizer: SGD, momentum=0.9, weight_decay=1e-5
# lr-schedule: step decay every 2 epochs
# M: 32
# beta: 5e-2

EPS = 1e-12

interpolate = partial(
    torch.nn.functional.interpolate,
    mode='bilinear',
    align_corners=False,
)

normalize = torch.nn.functional.normalize

###############################################################################
# Internal modules
###############################################################################

class BAP(torch.nn.Module):
    '''Bilinear Attention Pooling'''
    def __init__(self, pool='GAP'):
        super(BAP, self).__init__()
        assert pool in ['GAP', 'GMP']
        if pool == 'GAP':
            self.pool = None
        else:
            self.pool = torch.nn.AdaptiveMaxPool2d(1)

    def forward(self, features, attentions):
        B, C, H, W = features.size()
        _, M, AH, AW = attentions.size()

        # match size
        if AH != H or AW != W:
            attentions = interpolate(attentions, (H, W))

        # feature_matrix: (B, M, C) -> (B, M * C)
        if self.pool is None:
            feature_matrix = torch.einsum('imjk,injk->imn', (attentions, features))
            feature_matrix = feature_matrix.div(float(H * W)).view(B, -1)
        else:
            # This could be done without the loop
            feature_matrix = []
            for i in range(M):
                AiF = self.pool(features * attentions[:, i:i + 1, ...]).view(B, -1)
                feature_matrix.append(AiF)
            feature_matrix = torch.cat(feature_matrix, dim=1)

        # sign-sqrt
        s = feature_matrix.sign()
        feature_matrix =  s * feature_matrix.abs().add(EPS).sqrt()

        # l2 normalization along dimension M and C
        feature_matrix = normalize(feature_matrix, dim=-1)
        return feature_matrix


class ConvBNReLU(torch.nn.Module):
    '''Conv -> BatchNorm -> ReLU'''
    def __init__(self, inc, outc, ks=1, **kwargs):
        super().__init__()
        self.conv = torch.nn.Conv2d(inc, outc, ks, bias=False, **kwargs)
        self.bn = torch.nn.BatchNorm2d(outc, eps=0.001)

    def forward(self, x):
        x = self.bn(self.conv(x)) 
        return torch.nn.functional.relu(x, inplace=True)


##############################################################################
# Lightning Module for WSDAN
##############################################################################

class WSDAN(ImageClassifier):
    def __init__(
        self,
        num_attn: int=32,
        beta: float=5e-2,
        base_conf: Optional[dict]=None,
        model_conf: Optional[dict]=None,
    ):
        self.num_attn = num_attn
        self.beta = beta
        # parent class initialization
        ImageClassifier.__init__(self, base_conf=base_conf, model_conf=model_conf)

    def inject_backbone_args(self):
        self.model_conf.model_kw.update(
            features_only=True,
            out_indices=[4],
        )

    def setup_model(self):
        with torch.no_grad():
            out = self.backbone(torch.ones(1, 3, 33, 33))[-1]
            channels = out.shape[1]

        self.attentions = ConvBNReLU(channels, self.num_attn)
        self.bap = BAP(pool='GAP')
        self.fc = torch.nn.Linear(self.num_attn * channels, self.num_classes, bias=False)

        center = torch.zeros(self.num_classes, self.num_attn * channels)
        self.register_buffer('feature_center', center)

    def setup_metrics(self):
        super().setup_metrics()
        self.crop_accuracy = Accuracy('multiclass', num_classes=self.num_classes)
        self.drop_accuracy = Accuracy('multiclass', num_classes=self.num_classes)

    def forward(self, x, p_only=False):
        features = self.backbone(x)[-1]
        attentions = self.attentions(features)
        feature_matrix = self.bap(features, attentions)

        # This mystical multiply by 100 is in the original tensorflow implementation, as well as the pytorch
        # reimplementations. No explenation is provided. I tried removing it, but training doesn't work without it...
        # p = self.fc(feature_matrix * 100.0)
        p = self.fc(feature_matrix)

        if p_only: return p

        if self.training:
            weights = attentions.detach().sum((2, 3)).add_(EPS).sqrt_()
            weights = normalize(weights, p=1, dim=1, out=weights)
            k_idx = torch.multinomial(weights, 2, replacement=True)
            b_idx = torch.arange(k_idx.shape[0], device=k_idx.device)
            attention_map = attentions[b_idx[:,None], k_idx].contiguous()
        else:
            attention_map = attentions.mean(1, keepdim=True)

        return p, feature_matrix, attention_map

    @staticmethod
    def batch_crop(images, attention_map, theta=(0.4,0.6), pad_ratio=0.1):
        bs, _, h, w = images.size()
        crop_images = []
        ph = pad_ratio * h
        pw = pad_ratio * w
        for i in range(bs):
            atten_map = attention_map[i:i+1]
            if isinstance(theta, tuple):
                theta = random.uniform(*theta)
            theta_c = theta * atten_map.max()

            crop_mask = interpolate(atten_map, (h, w)).ge(theta_c)
            nz_inds = crop_mask[0, 0].nonzero(as_tuple=False)
            hmin = max(int(nz_inds[:, 0].min().item() - ph), 0)
            hmax = min(int(nz_inds[:, 0].max().item() + ph), h)
            wmin = max(int(nz_inds[:, 1].min().item() - pw), 0)
            wmax = min(int(nz_inds[:, 1].max().item() + pw), w)

            crop_images.append(
                interpolate(images[i:i+1,:, hmin:hmax, wmin:wmax], (h, w)))
        crop_images = torch.cat(crop_images, dim=0)
        return crop_images
        
    @staticmethod
    def batch_drop(images, attention_map, theta=(0.2, 0.5)):
        bs, _, h, w = images.size()
        drop_masks = []
        for i in range(bs):
            atten_map = attention_map[i:i+1]
            if isinstance(theta, tuple):
                theta = random.uniform(*theta)
            theta_d = theta * atten_map.max()

            drop_masks.append(interpolate(atten_map, (h, w)).lt(theta_d))
        drop_masks = torch.cat(drop_masks, dim=0)
        drop_images = images * drop_masks.float()
        return drop_images

    @staticmethod
    def centerloss(features, centers):
        # loss = features.sub(centers).square().sum(-1).mean()
        loss = torch.nn.functional.mse_loss(features, centers, reduction='sum') / centers.shape[0]
        return loss

    def training_step(self, batch, batch_index):
        x, y = batch

        # full-image predictions
        logits_raw, feature_mat, attention_map = self(x)

        # update class feature centers
        feature_center_batch = normalize(self.feature_center[y], dim=-1)
        center_update = self.beta * (feature_mat.detach() - feature_center_batch)
        # self.feature_center[y].add_(center_update)
        idx = y[:, None].expand_as(center_update)
        self.feature_center.scatter_add_(dim=0, index=idx, src=center_update)

        # attention crop predictions: first channel of attention_map is for crop
        with torch.no_grad():
            crops = self.batch_crop(x, attention_map[:,:1])
        logits_crop = self(crops, p_only=True)

        # attention drop predictions: second channel of attention_map is for drop
        with torch.no_grad():
            drops = self.batch_drop(x, attention_map[:,1:])
        logits_drop = self(drops, p_only=True)

        # calculate loss
        # loss = (self.objective(logits_raw, y) + self.objective(logits_crop, y) + self.objective(logits_drop, y))
        # loss = (1. / 3) * sum(self.objective(x, y) for x in (logits_raw, logits_crop, logits_drop))
        obj = self.objective
        loss = (1. / 3) * (obj(logits_raw, y) + obj(logits_crop, y) + obj(logits_drop, y))
        loss = loss + self.centerloss(feature_mat, feature_center_batch)

        # calculate accuracies
        acc = self.train_accuracy(logits_raw, y)
        crop_acc = self.crop_accuracy(logits_crop, y)
        drop_acc = self.drop_accuracy(logits_drop, y)

        # log
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/acc', acc, prog_bar=True)
        self.log('train/acc_crop', crop_acc, prog_bar=False)
        self.log('train/acc_drop', drop_acc, prog_bar=False)

        self.log('center_norm', self.feature_center.norm(2, dim=1).max(), prog_bar=True)

        return loss

    def inference_step(self, x):
        # full-image predictions
        logits_raw, _, attention_map = self(x)
        # attention crop predictions
        crops = self.batch_crop(x, attention_map[:,:1])
        logits_crop = self(crops, p_only=True)
        # combined predictions
        # logits = 0.5 * (logits_raw + logits_crop)
        logits = (logits_raw.softmax(1) + logits_crop.softmax(1)).mul_(0.5).log_()
        return logits

    def validation_step(self, batch, batch_index):
        x, y = batch
        logits = self.inference_step(x)

        # calculate loss and accuracy
        loss = self.objective(logits, y)
        acc = self.val_accuracy(logits, y)

        # log
        log_kw = dict(on_step=False, on_epoch=True)
        self.log('val/loss', loss, prog_bar=True, **log_kw)
        self.log('val/acc', acc, prog_bar=True, **log_kw)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.inference_step(x)
        return {'logits': logits, 'y': y}