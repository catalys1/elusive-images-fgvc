'''
Progressive Multi-Granularity Training, from "Fine-Grained Visual Classification
via Progressive Multi-Granularity Training of Jigsaw Patches" 
(https://arxiv.org/abs/2003.03836).

Adapted from https://github.com/PRIS-CV/PMG-Progressive-Multi-Granularity-Training
'''
import pytorch_lightning as pl
import torch, torchvision

from .base import BaseModule


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


def forward_func(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x1 = self.maxpool(x)
    x2 = self.layer1(x1)
    x3 = self.layer2(x2)
    x4 = self.layer3(x3)
    x5 = self.layer4(x4)
    return x3, x4, x5


################################################################################
# Lightning Module for PMG
################################################################################

class PMG(BaseModule):
    def __init__(
        self,
        feature_size: int=512,
        **kwargs,
    ):
        BaseModule.__init__(self, **kwargs)
        self.feature_size = feature_size
    
        # nclass = self.nclass
        # base = self.hparams.net
        # pretrained = self.hparams.pretrained

        # net = getattr(torchvision.models, base)(pretrained)
        # self.net = basemodel.ResnetBase(net, forward_func)
        self.maxpool = torch.nn.AdaptiveMaxPool2d((1,1))
        
        # conv blocks
        self.conv_blocks = torch.nn.ModuleList([
            ConvBlock(self.net.cout//c, feature_size, self.net.cout//2)
            for c in [4,2,1]
        ])

        # classifier blocks
        self.classifiers = torch.nn.ModuleList([
            Classifier(self.net.cout//2, feature_size, nclass)
            for _ in range(3)
        ])
        self.classifier_concat = Classifier(
            3*(self.net.cout//2), feature_size, nclass
        )

    def forward(self, x, level=None):
        # level can be 0, 1, 2, 3, or None (meaning all), specifying 
        # which of the intermediate outputs to compute and return
        layer_outs = self.net(x)

        # first 3 levels correspond to one of the granularity classifiers
        if level in (0,1,2):
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
    def jigsaw_generator(images, n):
        b, c, h, w = images.shape
        hn, wn = h//n, w//n
        s1 = [b, c, n, hn, n, wn]
        s2 = [b, c, n**2, hn, wn]
        s3 = [b, c, n, n, hn, wn]
        p = [0, 1, 2, 4, 3, 5]

        idx = torch.multinomial(torch.ones(n**2, device=images.device), n**2)
        jigsaw = images.view(s1).permute(p).reshape(s2)[:,:,idx]
        jigsaw = jigsaw.reshape(s3).permute(p).reshape(images.shape)

        return jigsaw.contiguous()

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        x, y = batch
        losses = []

        ns = [8, 4, 2, 1]
        for i in range(4):
            n = ns[i]
            js = self.jigsaw_generator(x, n) if n > 1 else x
            v = self(js, level=i)
            loss = self.lossfn(v, y)
            self.manual_backward(loss, opt)
            opt.step()
            opt.zero_grad()
            losses.append(loss.detach())

        acc = self.calc_accuracy(v, y)
        total_loss = sum(losses)
        losses = {f'train_loss_lvl_{i}': ll for i, ll in enumerate(losses)}

        log_kw = dict(on_step=False, on_epoch=True)
        self.log('train_loss', total_loss, prog_bar=True, **log_kw)
        self.log('train_acc', acc, prog_bar=True, **log_kw)
        self.log_dict(losses, **log_kw)

        return total_loss.detach()
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        outs = self(x, level=None)
        logits = sum(outs)
        loss = self.lossfn(logits, y)
        acc = self.calc_accuracy(logits, y)

        log_kw = dict(on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_loss', loss, prog_bar=True, **log_kw)
        self.log('val_acc', acc, prog_bar=True, **log_kw)
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        outs = self(x, level=None)
        logits = sum(outs)
        return {'logits':logits, 'y': y}
    