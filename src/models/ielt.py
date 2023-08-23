from copy import deepcopy
import math
from typing import Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT

import torch
from torch.nn.functional import normalize

from .base import ImageClassifier


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

    def get_saved_attention_weights(self, clear: bool=True):
        weights = self.saved_attn_weights
        if clear:
            self.saved_attn_weights = None
        return weights

    def forward(self, x):
        x, saved_attn = self.forward_layer(self.layer, x)
        self.saved_attn_weights = saved_attn
        return x


class SelectionBlock(torch.nn.Module):
    def __init__(self, layer, num_selections, patch_selector):
        super().__init__()
        self.num_selections = num_selections
        self.patch_selector = patch_selector

        self.layer = layer
        self.layer.attn = ExtractableAttentionWrapper(self.layer.attn)

    def get_complements(self, clear: bool=True):
        complements = self.complements
        if clear:
            self.complements = None
        return complements

    def forward(self, x: torch.Tensor):
        x = self.layer(x)
        weights = self.layer.attn.get_saved_attention_weights()

        select_idx = self.patch_selector(weights, self.num_selections)
        complements = x.gather(1, select_idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))

        self.complements = complements

        return x


class MultiHeadVoting(torch.nn.Module):
    def __init__(self, votes_per_head=24):
        super(MultiHeadVoting, self).__init__()

        self.votes_per_head = votes_per_head
        kernel = torch.tensor([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]], dtype=torch.float32)
        self.register_buffer('kernel', kernel)

    def enhance_local(self, count: torch.Tensor):
        b = count.shape[0]
        s = int(math.sqrt(count.shape[1]))
        count = count.reshape(b, 1, s, s)
        count = torch.nn.functional.conv2d(count, self.kernel, padding=1).flatten(1)
        return count

    def forward(self, x: torch.Tensor, select_num=None, last=False):
        B, patch_num = x.shape[0], x.shape[3] - 1
        select_num = self.votes_per_head if select_num is None else select_num
        score = x[:, :, 0, 1:]  # (Batch, Heads, Patches)

        _, select = score.topk(self.votes_per_head, dim=-1)
        select = select.flatten(1)  # (Batch, Heads * votes)

        # number of votes for each patch
        count = x.new_zeros(B, patch_num)
        count.scatter_add_(dim=1, index=select, src=x.new_ones(1, 1).expand_as(select))

        if not last:
            count = self.enhance_local(count)

        _, patch_idx = count.topk(select_num, dim=-1)
        patch_idx += 1
        return patch_idx


class CrossLayerRefinement(torch.nn.Module):
    def __init__(self, layer, hidden_size):
        super().__init__()
        self.layer = layer
        self.norm = torch.nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, x):
        x = self.layer(x)
        x = self.norm(x)
        weights = self.layer.attn.get_saved_attention_weights()
        return x, weights


class IELT(ImageClassifier):
    def __init__(
        self,
        reinit_final_blocks: bool=True,
        votes_per_head: int=24,
        total_selections: int=126,
        warmup_epochs: int=10,
        alpha: float=0.4,
        drop_rate: float=0.0,
        base_conf: Optional[dict]=None,
        model_conf: Optional[dict]=None,
    ):
        self.reinit_final_blocks = reinit_final_blocks
        self.votes_per_head = votes_per_head
        self.total_selections = total_selections
        self.warmup_epochs = warmup_epochs
        self.alpha = alpha
        self.drop_rate = drop_rate

        # this gets set to True after a warmup period
        self.updating_layer_selections = False

        super().__init__(base_conf=base_conf, model_conf=model_conf)

    def inject_backbone_args(self):
        self.model_conf.model_kw.update(
            global_pool='',
        )

    def setup_model(self):
        blocks = self.backbone.blocks

        refine_block = blocks[-1]
        key_block = deepcopy(blocks[-1])
        # in the original code, the refine and key blocks are randomly initialized except, apparently,
        # when training on "dogs" or "nabrids" (yes, spelled incorrectly in the code, so it probably
        # didn't actually get applied in that case); in those 2 special cases, the blocks used the
        # pretrained weights from the final transformer block.
        if self.reinit_final_blocks:
            self._init_block(refine_block)
            self._init_block(key_block)

        refine_block.attn = ExtractableAttentionWrapper(refine_block.attn)
        # key_block.attn = ExtractableAttentionWrapper(key_block.attn)

        select_rate = torch.tensor([16.0, 14, 12, 10, 8, 6, 8, 10, 12, 14, 16]) / self.total_selections
        num_selects = select_rate.mul(self.total_selections).long()
        self.register_buffer('select_rate', select_rate)
        self.register_buffer('num_selects', num_selects)

        self.selector = MultiHeadVoting(self.votes_per_head)

        # replace the normal transformer blocks with special SelectionBlocks
        self.backbone.blocks = torch.nn.Sequential(*[
            SelectionBlock(b, ns, self.selector) for b, ns in zip(blocks[:-1], self.num_selects)
        ])
        # we don't need these in the backbone
        self.backbone.norm = torch.nn.Identity()
        self.backbone.head = torch.nn.Identity()

        self.cross_layer_refinement = CrossLayerRefinement(refine_block, self.backbone.embed_dim)
        self.key_block = key_block
        self.key_norm = torch.nn.LayerNorm(self.backbone.embed_dim, eps=1e-6)

        self.head = torch.nn.Sequential(
            torch.nn.Dropout(self.drop_rate),
            torch.nn.Linear(self.backbone.embed_dim, self.num_classes),
        )

    def _init_block(self, block: torch.nn.Module):
        for m in block.modules():
            if isinstance(m, (torch.nn.Linear, torch.nn.LayerNorm)):
                torch.nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)


    @torch.no_grad()
    def _update_layer_select(self, sort_idx: torch.Tensor):
        n = self.num_selects.nelement()
        bounds = self.num_selects.cumsum(-1)
        counts = torch.bucketize(sort_idx.flatten(), bounds, right=True).clamp_max(n - 1).bincount(minlength=n)

        alpha = 1e-3
        new_rate = normalize(counts.float(), p=1, dim=-1)
        self.select_rate = normalize(torch.lerp(self.select_rate, new_rate, alpha), p=1, dim=-1)
        self.num_selects = self.select_rate.mul(self.total_selections).round().long()

        # update each of the SelectionBlocks, since they store their num_selections
        for block, n in zip(self.backbone.blocks, self.num_selects):
            block.num_selections = n

    def forward(self, x: torch.Tensor):
        x = self.backbone.forward_features(x)
        cls_token = x[:, :1]
        complements = torch.cat([cls_token] + [b.get_complements() for b in self.backbone.blocks], dim=1)
        clr, weights = self.cross_layer_refinement(complements)
        sort_idx = self.selector(weights, select_num=24, last=True)

        if self.updating_layer_selections:
            self._update_layer_select(sort_idx)

        x = clr.gather(1, index=sort_idx.unsqueeze(-1).expand(-1, -1, clr.size(-1)))
        x = torch.cat((cls_token, x), dim=1)
        x = self.key_block(x)
        x = self.key_norm(x)

        comp_logits = self.head(clr[:, 0])
        probs = comp_logits.softmax(dim=-1)
        assist_logits = probs * self.head[1].weight.sum(-1)
        part_logits = self.head(x[:, 0]) + assist_logits

        return part_logits, comp_logits

    def on_train_epoch_start(self):
        if self.trainer.current_epoch >= self.warmup_epochs:
            self.updating_layer_selections = True

    def on_validation_epoch_start(self):
        self.updating_layer_selections = False

    def step(self, batch, accuracy_metric):
        x, y = batch
        part_logits, complement_logits = self(x)
        loss_p = self.objective(part_logits, y)
        loss_c = self.objective(complement_logits, y)
        loss = torch.lerp(loss_p, loss_c, self.alpha)
        acc = accuracy_metric(part_logits, y)
        return part_logits, loss, acc

    def training_step(self, batch, batch_idx):
        logits, loss, acc = self.step(batch, self.train_accuracy)

        self.log('train/loss', loss, prog_bar=True)
        self.log('train/acc', acc, prog_bar=True)

        return {'loss': loss, 'pred': logits}

    def validation_step(self, batch, batch_idx):
        logits, loss, acc = self.step(batch, self.val_accuracy)

        self.log('val/loss', loss, prog_bar=True, sync_dist=True)
        self.log('val/acc', acc, prog_bar=True, sync_dist=True)

        return {'loss': loss, 'pred': logits}

    def predict_step(self, batch, batch_idx):
        x, y = batch
        pred, _ = self(x)

        self.predictions.append(pred.detach().cpu())
        self.labels.append(y.cpu())

        return pred