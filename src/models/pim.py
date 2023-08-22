# adapted from https://github.com/chou141253/FGVC-PIM/
import copy # TODO: needed for deepcopy
from typing import Dict, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT

import torch
from torchvision.models.feature_extraction import create_feature_extractor

from .base import ImageClassifier


def default_feature_layers(name: str):
    if name.startswith('vit'):
        return {
            'blocks.8': 'layer1',
            'blocks.9': 'layer2',
            'blocks.10': 'layer3',
            'blocks.11': 'layer4',
        }
    if 'resnet50' in name:
        return {
            'layer1.2.act3': 'layer1',
            'layer2.3.act3': 'layer2',
            'layer3.5.act3': 'layer3',
            'layer4.2.act3': 'layer4',
        }

    raise RuntimeError(f'{name} not currently supported')


################################################################################
# Internal Modules
################################################################################

class FeaturePyramid(torch.nn.Module):
    def __init__(self, inputs, fpn_size):
        super().__init__()

        inp_names = [name for name in inputs]
        self.fpn_size = fpn_size

        self.projections = torch.nn.ModuleDict()
        self.upsamples = torch.nn.ModuleDict()

        for i, (name, val) in enumerate(inputs.items()):
            if val.ndim == 3:  # transformer
                in_dim = val.size(-1)
                proj = torch.nn.Sequential(
                    torch.nn.Linear(in_dim, in_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(in_dim, self.fpn_size),
                )
            elif val.ndim == 4:  # convnet
                in_dim = val.size(1)
                proj = torch.nn.Sequential(
                    torch.nn.Conv2d(in_dim, in_dim, 1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(in_dim, self.fpn_size, 1)
                )
            self.projections[name] = proj
            
            if i != 0:
                if val.ndim == 3:  # transformer
                    in_dim = val.size(1)
                    out_dim = inputs[inp_names[i - 1]].size(1)
                    if in_dim != out_dim:
                        upsample = torch.nn.Conv1d(in_dim, out_dim, 1) # for spatial domain
                    else:
                        upsample = torch.nn.Identity()
                elif val.ndim == 4:  # convnet
                    upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                self.upsamples[name] = upsample

    def forward(self, x: Dict[str, torch.Tensor]):
        hs = []
        for i, name in enumerate(x):
            x[name] = self.projections[name](x[name])
            hs.append(name)

        for i in range(len(hs) - 1, 0, -1):
            x1_name = hs[i]
            x0_name = hs[i - 1]
            # upsample and add
            x[x0_name] = x[x0_name] + self.upsamples[x1_name](x[x1_name])

        return x


class Selector(torch.nn.Module) :
    def __init__(self, num_select: dict):
        super().__init__()

        self.num_select = num_select
    
    def forward(self, x: Dict[str, torch.Tensor], logits: Dict[str, torch.Tensor]):
        logits['select'] = []
        logits['drop'] = []
        selections = {}
        for name, feats in x.items():
            if feats.ndim == 4:
                feats = feats.flatten(2).transpose(1, 2)
            n = feats.size(-1)
            
            logit = logits[name]
            c = logit.size(-1)
            
            num_select = self.num_select[name]
            probs = torch.softmax(logits[name], dim=-1)
            ranks = probs.max(dim=-1)[0].argsort(dim=-1, descending=True)

            top_rank = ranks[:, :num_select, None]
            bot_rank = ranks[:, num_select:, None]
            sf = feats.gather(1, top_rank.expand(-1, -1, n))
            preds_1 = logit.gather(1, top_rank.expand(-1, -1, c))
            preds_0 = logit.gather(1, bot_rank.expand(-1, -1, c))

            selections[name] = sf

            logits['select'].append(preds_1)
            logits['drop'].append(preds_0)

        return selections


class Combiner(torch.nn.Module):
    def __init__(self, total_num_selects, num_classes, proj_dim, drop_rate=0.1):
        super().__init__()

        # build one layer structure (with adaptive module)
        self.proj_dim = proj_dim
        num_joints = total_num_selects // 32

        self.param_pool0 = torch.nn.Linear(total_num_selects, num_joints)

        self.adj1 = torch.nn.Parameter(torch.eye(num_joints).add(1).mul(1 / 100))
        self.conv1 = torch.nn.Conv1d(self.proj_dim, self.proj_dim, 1)
        self.batch_norm1 = torch.nn.BatchNorm1d(self.proj_dim)
        
        self.conv_q1 = torch.nn.Conv1d(self.proj_dim, self.proj_dim // 4, 1)
        self.conv_k1 = torch.nn.Conv1d(self.proj_dim, self.proj_dim // 4, 1)
        self.alpha1 = torch.nn.Parameter(torch.zeros(1))

        # merge information
        self.param_pool1 = torch.nn.Linear(num_joints, 1)
        
        # class predict
        self.dropout = torch.nn.Dropout(p=drop_rate)
        self.classifier = torch.nn.Linear(self.proj_dim, num_classes)

        self.tanh = torch.nn.Tanh()

    def forward(self, x: Dict[str, torch.Tensor]):
        hs = []
        for name in x:
            hs.append(x[name])
        
        hs = torch.cat(hs, dim=1).transpose(1, 2).contiguous()
        hs = self.param_pool0(hs)

        # adaptive adjacency
        q1 = self.conv_q1(hs).mean(1)
        k1 = self.conv_k1(hs).mean(1)
        A1 = self.tanh(q1.unsqueeze(-1) - k1.unsqueeze(1))
        A1 = self.adj1 + A1 * self.alpha1

        # graph convolution
        hs = self.conv1(hs)
        hs = torch.matmul(hs, A1)
        hs = self.batch_norm1(hs)

        # predict
        hs = self.param_pool1(hs)
        hs = self.dropout(hs)
        hs = hs.flatten(1)
        hs = self.classifier(hs)

        return hs


################################################################################
# Lightning Module for PIM
################################################################################

class PIM(ImageClassifier):
    def __init__(
        self,
        feature_size: int=512,
        img_size: int=224,
        fpn_size: int=512,
        num_selects: Optional[dict]=None,
        return_nodes: Optional[dict]=None,
        classifier_drop_rate: float=0.1,
        lambda_b: float=0.5,
        lambda_c: float=1.0,
        lambda_n: float=5.0,
        base_conf: Optional[dict]=None,
        model_conf: Optional[dict]=None
    ):
        self.feature_size = feature_size    
        self.img_size = img_size
        self.fpn_size = fpn_size

        if num_selects is None:
            num_selects = {
                'layer1': 32,
                'layer2': 32,
                'layer3': 32,
                'layer4': 32
            }
        self.num_selects = num_selects
        
        self.return_nodes = return_nodes

        self.classifier_drop_rate = classifier_drop_rate

        self.lambda_b = lambda_b
        self.lambda_c = lambda_c
        self.lambda_n = lambda_n

        ImageClassifier.__init__(self, base_conf=base_conf, model_conf=model_conf)
    
    def setup_model(self):
        if self.return_nodes is None:
            self.return_nodes = default_feature_layers(self.model_conf.model_name)

        self.layers = list(self.return_nodes.values())

        # turn backbone into feature extractor
        self.backbone = create_feature_extractor(self.backbone, return_nodes=self.return_nodes)
        rand_in = torch.randn(1, 3, self.img_size, self.img_size)
        dummy_x = self.backbone(rand_in)

        # feature pyramid network
        self.fpn = FeaturePyramid(dummy_x, self.fpn_size)
        self.fpn_classifiers = torch.nn.ModuleDict()
        for name in dummy_x:
            m = torch.nn.Sequential(
                torch.nn.Conv1d(self.fpn_size, self.fpn_size, 1),
                torch.nn.BatchNorm1d(self.fpn_size),
                torch.nn.ReLU(),
                torch.nn.Conv1d(self.fpn_size, self.num_classes, 1)
            )
            self.fpn_classifiers[name] = m

        # selector
        self.selector = Selector(self.num_selects)

        # combiner
        total_num_selects = sum(self.num_selects.values())
        self.combiner = Combiner(total_num_selects, self.num_classes, self.fpn_size, self.classifier_drop_rate)

    def forward(self, x: torch.Tensor):
        logits = {}
        x = self.backbone(x)
        x = self.fpn(x)

        # predict for each feature point
        for name, feats in x.items():
            if feats.ndim == 3:
                logit = feats.transpose(1, 2)
            elif feats.ndim == 4:
                logit = feats.flatten(2)
            logits[name] = self.fpn_classifiers[name](logit).transpose(1, 2).contiguous()

        selected = self.selector(x, logits)
        combined = self.combiner(selected)
        logits['combined'] = combined
        
        return logits

    def training_step(self, batch, batch_idx: int):
        x, y = batch
        outputs = self(x)

        # calculate loss
        loss_layers = 0
        for name in self.layers:
            loss_layers = loss_layers + torch.nn.functional.cross_entropy(outputs[name].mean(1), y)

        loss_drop = 0
        drop_label = torch.full((1, 1), -1.0, device=x.device)
        for v in outputs['drop']:
            v = v.flatten(0, 1).tanh()
            loss_drop = loss_drop + torch.nn.functional.mse_loss(v, drop_label.expand_as(v))

        loss_combined = torch.nn.functional.cross_entropy(outputs['combined'], y)

        loss = loss_drop * self.lambda_n + loss_layers * self.lambda_b + loss_combined * self.lambda_c
        accuracy = self.train_accuracy(outputs['combined'], y)

        self.log('train/loss', loss, prog_bar=True)
        self.log('train/acc', accuracy, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx: int):
        x, y = batch
        outputs = self(x)

        loss = torch.nn.functional.cross_entropy(outputs['combined'], y)
        accuracy = self.val_accuracy(outputs['combined'], y)

        self.log('val/loss', loss, prog_bar=True)
        self.log('val/acc', accuracy, prog_bar=True)

        return loss