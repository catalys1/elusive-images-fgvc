'''
SIM-Trans, from "SIM-Trans: Structure Information Modeling Transformer for Fine-grained Visual Categorization"
(https://arxiv.org/abs/2208.14607).

Parts of this code are adapted directly from https://github.com/PKU-ICST-MIPL/SIM-Trans_ACMMM2022
'''
from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT

import timm
import torch

from .base import ImageClassifier


ACT2FN = {
    'gelu': torch.nn.functional.gelu,
    'relu': torch.nn.functional.relu,
}


class GraphConvolution(torch.nn.Module):
    '''Graph convolution layer.'''
    def __init__(self, in_features, out_features, bias=False, dropout = 0.1):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.zeros(in_features, out_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        self.relu = torch.nn.LeakyReLU(0.2)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / self.weight.shape[1]**0.5
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        weight = self.weight.float()
        support = torch.matmul(input, weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return self.dropout(self.relu(output + self.bias))
        else:
            return self.dropout(self.relu(output))

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
    

class GCN(torch.nn.Module):
    '''Graph convolution network with two layers.'''
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = torch.nn.functional.relu(self.gc1(x, adj))
        x = torch.nn.functional.dropout(x, self.dropout)
        x = self.gc2(x, adj)
        return x


class ExtractableAttentionWrapper(torch.nn.Module):
    '''Slight modification to the original attention layer forward pass to save the attention weights so they
    can be retrieved later. Also, in order to get the attention weights, we can't use pytorch fused attention operator.
    '''
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    @staticmethod
    def forward_layer(self, x):
        # NOTE: `self` in this case is actually `self.layer`, since this is a staticmethod and we pass in `self.layer`
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        ### Added this to get access to attention weights
        saved_attn = attn.clone()
        ###
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        # return the saved attention matrix also
        return x, saved_attn

    def forward(self, x):
        x, saved_attn = self.forward_layer(self.layer, x)
        self.saved_attn_weights = saved_attn
        return x


class RelativeCoordPredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        N, C, H, W = x.shape
        HW = H * W
        size = H

        # sum attention scores over attention heads
        mask = torch.sum(x, dim=1)

        mask = mask.view(N, HW)
        thresholds = torch.mean(mask, dim=1, keepdim=True)
        binary_mask = mask.gt(thresholds).float().view(N, H, W)

        masked_x = x * binary_mask[:, None]
        masked_x = masked_x.view(N, C, HW).transpose(1, 2).contiguous()  # (N, S, C)
        reduced_x_max_index = masked_x.mean(dim=-1).argmax(dim=-1)

        index = torch.arange(N, device=x.device)

        label = self.build_basic_label(size, x.device)

        # Build Label
        label = label.unsqueeze(0).expand((N, H, W, 2)).view(N, HW, 2)  # (N, S, 2)
        basic_anchor = label[index, reduced_x_max_index, :].unsqueeze(1)  # (N, 1, 2)
        relative_coord = label.sub(basic_anchor).mul(1 / size)
        relative_dist = relative_coord.norm(p=2, dim=-1)
        relative_angle = torch.atan2(relative_coord[:, :, 1], relative_coord[:, :, 0])  # (N, S) in (-pi, pi)
        relative_angle = relative_angle.mul(0.5 / torch.pi).add(0.5)  # (N, S) in (0, 1)

        binary_relative_mask = binary_mask.view(N, HW)
        relative_dist = relative_dist * binary_relative_mask
        relative_angle = relative_angle * binary_relative_mask

        basic_anchor = basic_anchor.squeeze(1)  # (N, 2)

        relative_coord_total = torch.cat((relative_dist.unsqueeze(2), relative_angle.unsqueeze(2)), dim=-1)

        # average over attention heads
        position_weight = masked_x.mean(dim=-1, keepdim=True)
        position_weight = position_weight @ position_weight.transpose(1,2)

        return relative_coord_total, basic_anchor, position_weight, reduced_x_max_index
    
    def build_basic_label(self, size, device='cpu'):
        # basic_label = np.array([[(i, j) for j in range(size)] for i in range(size)])
        v = torch.arange(size, dtype=torch.float32, device=device)
        v = torch.stack(torch.meshgrid(v, v, indexing='ij'), -1)
        return v


class PartStructureLayer(torch.nn.Module):
    '''Part structure layer. Implements Structure Information Learning from the paper
    (see Section 3.2). Replaces the Part_Structure class from the original implementation,
    and incorporates the modifications they made to the Transformer forward pass.

    Args:
        layer (torch.nn.Module): attention layer (timm.models.vision_transformer.Attention).
        hidden_size (int): hidden dimension of the transformer model.
    '''
    def __init__(self, layer: torch.nn.Module, hidden_size: int):
        super().__init__()

        self.layer = layer
        self.layer.attn = ExtractableAttentionWrapper(self.layer.attn)
        self.relative_coord_predictor = RelativeCoordPredictor()
        self.gcn = GCN(2, 512, hidden_size, dropout=0.1)

    def part_attention(self, attn):
        '''Replaces the Part_Attention class from the original code.'''
        # attention between CLS token and all other tokens
        attn = attn[:, :, 0, 1:]
        max_value, max_idx = attn.max(2)

        B, C, num_patch = attn.shape

        H = int(num_patch**0.5)
        attention_map = attn.view(B, C, H, H)

        return attention_map, attn, max_idx, max_value
    
    def part_structure(self, attention_map):
        B, C, H, W = attention_map.shape
        structure_info, anchor, position_weight, _ = self.relative_coord_predictor(attention_map)
        structure_info = self.gcn(structure_info, position_weight)

        idx = (anchor[:, 0] * H + anchor[:, 1]).long()
        idx = idx[:, None, None].expand(-1, 1, structure_info.shape[2])

        return structure_info.gather(1, idx).squeeze()

    def forward(self, hidden_states):
        # ViT attention layer
        hidden_states = self.layer(hidden_states)
        attn_weights = self.layer.attn.saved_attn_weights
        self.layer.attn.saved_attn_weights = None

        # part selection
        attn_map = self.part_attention(attn_weights)[0]

        # structure modeling
        structure = self.part_structure(attn_map)

        # adding structure information to the CLS token
        hidden_states[:, 0] += structure

        # save the CLS token state
        self.cls_token = hidden_states[:, 0].clone()

        return hidden_states


class OverlapPatchEmbedding(torch.nn.Module):
    def __init__(
        self,
        img_size: int=224,
        patch_size: int=16,
        stride: int=12,
        in_chans: int=3,
        embed_dim: int=768,
        flatten: bool=True,
        bias: bool=True,
    ):
        super().__init__()
        img_size = timm.layers.helpers.to_2tuple(img_size)
        patch_size = timm.layers.helpers.to_2tuple(patch_size)

        (h, w), (ph, pw) = img_size, patch_size
        self.grid_size = (int((h - ph) / stride + 1), int((w - pw) / stride + 1))
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = torch.nn.Conv2d(in_chans, embed_dim, patch_size, stride, bias=bias)
        self.flatten = flatten

    def forward(self, x: torch.Tensor):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        return x


class SIMTrans(ImageClassifier):
    def __init__(
        self,
        drop_rate: float=0.0,
        base_conf: Optional[dict]=None,
        model_conf: Optional[dict]=None,
    ):
        self.drop_rate = drop_rate
        # parent class initialization
        super().__init__(base_conf=base_conf, model_conf=model_conf)

    def inject_backbone_args(self):
        self.model_conf.model_kw.update(
            global_pool='',
            embed_layer=OverlapPatchEmbedding,
        )

    def setup_model(self):
        hidden_size = self.backbone.embed_dim
        # wrap the last 3 layers with PartStructureLayer
        for i in range(-3, 0):
            block = self.backbone.blocks[i]
            self.backbone.blocks[i] = PartStructureLayer(block, hidden_size)
        
        # remove the ViT classification head
        self.backbone.head = torch.nn.Identity()

        self.head = torch.nn.Sequential(
            torch.nn.BatchNorm1d(hidden_size * 3),
            torch.nn.Dropout(self.drop_rate),
            torch.nn.Linear(hidden_size * 3, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ELU(inplace=True),
            torch.nn.Dropout(self.drop_rate),
            torch.nn.Linear(1024, self.model_conf.num_classes),
        )

    def forward(self, x: torch.Tensor):
        last_hidden_state = self.backbone.forward_features(x)

        # normalize and concatenate the CLS tokens from last three layers
        cls_toks = []
        for i in range(-3, 0):
            layer = self.backbone.blocks[i]
            if i < -1:
                cls_tok = self.backbone.norm(layer.cls_token)
                cls_toks.append(cls_tok)
            layer.cls_token = None
        # NOTE: the last layer has already been normalized
        cls_toks.append(last_hidden_state[:, 0])
        cls_toks = torch.cat(cls_toks, -1)

        logits = self.head(cls_toks)

        return logits, last_hidden_state

    def contrastive_loss(self, features: torch.Tensor, labels: torch.Tensor):
        margin = 0.3
        B = features.shape[0]

        pos_labels = labels.view(-1, 1).eq(labels)
        neg_labels = ~pos_labels
    
        features = torch.nn.functional.normalize(features)
        cos_matrix = features.mm(features.t())
        scores = (1 - cos_matrix) / 2.0

        mask = pos_labels.clone()
        mask[range(B), range(B)] = False
        positive_scores = torch.where(mask, scores, torch.zeros_like(scores[0, 0]))
        positive_scores = positive_scores.sum(1, keepdim=True) / mask.sum(1, keepdim=True).add(1e-6)

        neg_labels[(margin + positive_scores - scores) < 0] = 0

        loss = pos_labels.mul(1 - cos_matrix).sum() + neg_labels.mul(1 + cos_matrix).sum()
        loss /= (B * B)

        return loss

    def step(self, batch, accuracy_metric, add_contrastive=False):
        x, y = batch
        logits, last_hidden_state = self(x)
        loss = self.objective(logits, y)
        if add_contrastive:
            contrastive = self.contrastive_loss(last_hidden_state[:, 0], y)
            loss = loss + contrastive
        acc = accuracy_metric(logits, y)
        return logits, loss, acc

    def training_step(self, batch: Any, batch_idx: int):
        logits, loss, accuracy = self.step(batch, self.train_accuracy, add_contrastive=True)

        self.log('train/loss', loss, prog_bar=True, sync_dist=True)
        self.log('train/acc', accuracy, prog_bar=True, sync_dist=True)

        return {'loss': loss, 'pred': logits}

    def validation_step(self, batch: Any, batch_idx: int):
        logits, ce_loss, accuracy = self.step(batch, self.val_accuracy, add_contrastive=False)

        self.log('val/loss', ce_loss, prog_bar=True, sync_dist=True)
        self.log('val/acc', accuracy, prog_bar=True, sync_dist=True)

        return {'loss': ce_loss, 'pred': logits}
