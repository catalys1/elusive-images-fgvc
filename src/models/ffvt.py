from .base import ImageClassifier, get_backbone
import torch
from typing import Optional
import timm
# adapted from https://github.com/Markin-Wang/FFVT/tree/main 

# linear, crossentropyloss, dropout, softmax, conv2d, layernorm are imported models
# also resnet (?)
NUM_LAYERS = 1 # 12
HIDDEN_SIZE = 1 # 768
DROP_RATE = 0.1 # 0.0
MLP_DIM = 1 # 3072

# Attention layer adapted from https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
class newAttention(timm.models.vision_transformer.Attention):
    def forward(self,x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        q = q * self.scale
        attnsco = q @ k.transpose(-2, -1)
        attn = attnsco.softmax(dim=-1)
        weights = attn
        attn = self.attn_drop(attn)
        x = attn @ v # context layer
        # x = x.transpose(1, 2).reshape(B, N, C)
        # x = self.proj(x)
        # x = self.proj_drop(x)
        # return x
    
        # new code for attention
        # context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = x.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights, self.softmax2(attnsco)[:,:,:,0]

# multi-layer perceptron for processing tokens
class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = torch.nnLinear(HIDDEN_SIZE, MLP_DIM)
        self.fc2 = torch.nnLinear(HIDDEN_SIZE, MLP_DIM)
        self.act_fn = torch.nn.functional.gelu
        self.dropout = torch.nn.Dropout(DROP_RATE)

        self._init_weights()

    def _init_weights(self):
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.normal_(self.fc1.bias, std=1e-6)
        torch.nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Block(torch.nn.Module):
    def __init__(self):
        super(Block, self).__init__()
        self.attentnorm = torch.nn.LayerNorm(HIDDEN_SIZE,eps=1e-6)
        self.ffnorm = torch.nn.LayerNorm(HIDDEN_SIZE,eps=1e-6)
        self.mLP = MLP()
        self.attn = newAttention()
        return
    def forward(self, x):
        actualx = x
        x = self.attentnorm(x)
        x,weights,cont = self.attn(x)
        savex = x+actualx
        x = self.ffnorm(savex)
        x = self.mLP(x)
        x = savex+x
        return x, weights

# QUESTIONS
# standardize layer numbers, input sizes?

class Transformer(torch.nn.Module):
    def __init__(self, size):
        super(Transformer,self).__init__()
        # encoder layer
        self.layer = torch.nn.ModuleList([Block() for i in range(NUM_LAYERS-1)])
        self.maws = MAWS()
        self.lastlayer = Block()
        self.encoder = torch.nn.LayerNorm(HIDDEN_SIZE,eps=1e-6) 
    
    def forward(self, embeds):
        # send embedding through encoder, return x, weights
        tokens=[[] for i in range(embeds.shape[0])]
        weights = []
        # now forward function for encoder
        for layer_block in self.layer:
            # forward step
            embed,weight,cont = layer_block(embeds)
            weights.append(weight)

            # feature fusion
            num,inx = self.maws(weight,cont)
            for i in range(inx.shape[0]):
                tokens[i].extend(embed[i,inx[i,:num]])

        # more feature fusion
        tokens=[torch.stack(token) for token in tokens]
        tokens = torch.stack(tokens).squeeze(1)
        concat = torch.cat((embeds[:,0].unsqueeze(1), tokens), dim=1)
        # do last layer with selected 'important' tokens from MAWS
        ff_states, ff_weights, ff_contri = self.lastlayer(concat)
        encoded = self.encoder(ff_states)
        return encoded, weights # which would be list of weights from each layer

# mutual attention weight selection
class MAWS(torch.nn.Module):
    def __init__(self):
        super(MAWS, self).__init__()

    def forward(self, x, contributions):
        length = x.size()[1]
        contributions = contributions.mean(1)
        weights = x[:,:,0,:].mean(1)
        scores = contributions*weights
        max_inx = torch.argsort(scores, dim=1,descending=True)
        return None, max_inx  

class FFVT(ImageClassifier):
    def __init__(
        self,
        feature_size: int=512,
        base_conf: Optional[dict]=None,
        model_conf: Optional[dict]={'model_name':'vit_base_patch16_224_miil_in21k',"optimizer_name":'SGD'}
        # what is optim_kw?
    ):
        # training is done with gradient accumulation steps, so no extra code needed for optimizer
        self.feature_size = feature_size    

        # parent class initialization
        ImageClassifier.__init__(self, base_conf=base_conf, model_conf=model_conf)
    
    def setup_model(self):
        # need transformer and head
        self.transformer = Transformer(self.feature_size)
        self.linear = torch.nn.Linear(self.feature_size, self.num_classes)

    def forward(self, x):
        embeds = self.backbone(x)
        x, weights = self.transformer(embeds)
        logits = self.linear(x[:,0])
        # get x, weights from transformer
        # get logits/predictions from head (linear layer)
        # classification 
        # use cross entropy loss (with no label smoothing)
        # return loss, logits
        return logits # OR loss, weights??

# forward step:
# logits = self.head(x[:,0 ])
#  does embedding with input x and then encodes that, 
# gets attention weights with encoder
 # encoder has feature fusion step, self.ff_token_select = maws
 # last layer is block
 # encoder is layer norm
 # for each layer, feature fusion in encoder
