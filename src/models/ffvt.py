from .base import ImageClassifier, get_backbone
import torch
from typing import Optional
import timm
# adapted from https://github.com/Markin-Wang/FFVT/tree/main 

# TO RUN: 
# create the config file by merging several configs
# python utils/configs.py src/configs/trainer/test_trainer.yaml src/configs/models/resnet50.yaml src/configs/data/cub.yaml -f config.yaml
# "fit" tells Lightning to run the training loop
# srun python run.py fit -c config.yaml
# conda activate hp
# salloc -N 2 -n 8 --gpus 2 --mem 32G --time 0:30:00 --qos cs

# linear, crossentropyloss, dropout, softmax, conv2d, layernorm are imported models
# also resnet (?)
NUM_LAYERS = 2 # 12, 1 doesn't work for testing
HIDDEN_SIZE = 32 # 768
DROP_RATE = 0.1 # 0.0
MLP_DIM = 1 # 3072
NUM_TOKEN = 12

# Attention layer adapted from https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
class newAttention(timm.models.vision_transformer.Attention):
    # transformer gets embedding output
        # get patch embeddings from Conv2d layer
        # cls_token, line 182 on
    # passes to encoder to get encoded and attention weights
        # encoder passes through LayerNorm and then attention layer 
        # needs to have 4 dimensions when split into q,k,v

    def __init__(self,dim):
        super().__init__(dim)
        self.out = torch.nn.Linear(self.head_dim, self.head_dim)
        self.proj_dropout = self.proj_drop 
        self.softmax2 = torch.nn.Softmax(dim=-2)

    def forward(self,x):
        B, N , C= x.shape # only B, N, no C because diff. dimensions
        # shape to 200,32,3,8,4, from 19200 - not valid, need more data???
        # head dim is 4, 8 is num heads, 32 is dim  
        # only have 200*32*3 tensor
        # if you want num_heads and head_dim to be 1, headdim= dim//num_heads so 1 = 1/1 or 1 = 32/32
        # dim
        # qkv = self.qkv(x) # 200x96
        # qkv=qkv.reshape(B,N,3,1) # or 32,200,1,3 qkv.permute(2,0,3,1)
        # qkv = qkv.permute(2,0,3,1)
        # qkv = qkv.unbind(2)
        # print(f"len {len(qkv)}") # 200x96 is size (32*3) -> 32x200xnum_headsx3 - no head_dim
        # print(qkv[0].size())
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        q = q * self.scale
        attnsco = q @ k.transpose(-2, -1)
        attn = attnsco.softmax(dim=-1)
        weights = attn
        attn = self.attn_drop(attn)
        x = attn @ v # context layer
        #x = x.transpose(1, 2).reshape(B, N, C)
        # x = self.proj(x)
        # x = self.proj_drop(x)
        # return x
    
        # new code for attention
        # context_layer = torch.matmul(attention_probs, value_layer)
        all_head_size = self.num_heads * self.head_dim/self.num_heads
        context_layer = x.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.head_dim,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights, self.softmax2(attnsco)[:,:,:,0]

# multi-layer perceptron for processing tokens
class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(HIDDEN_SIZE, MLP_DIM)
        self.fc2 = torch.nn.Linear(HIDDEN_SIZE, MLP_DIM)
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
    def __init__(self,size):
        super(Block, self).__init__()
        self.attentnorm = torch.nn.LayerNorm(HIDDEN_SIZE,eps=1e-6)
        self.ffnorm = torch.nn.LayerNorm(HIDDEN_SIZE,eps=1e-6)
        self.mLP = MLP()
        self.attn = newAttention(HIDDEN_SIZE) # hidden_size = dim param
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

class NewBlock(torch.nn.Module):
    def __init__(self,layer,size):
        super(NewBlock, self).__init__()
        self.layer = layer
        self.maws = MAWS()
        self.attentnorm = torch.nn.LayerNorm(HIDDEN_SIZE,eps=1e-6)
        self.ffnorm = torch.nn.LayerNorm(HIDDEN_SIZE,eps=1e-6)
        self.mLP = MLP()
        self.attn = newAttention(HIDDEN_SIZE) # hidden_size = dim param
        self.weight = None
        return

    def forward(self, embeds):
        tokens = [[] for i in range(embeds.shape[0])]
        x = embeds
        actualx = x
        x = self.attentnorm(x)
        x,weights,cont = self.attn(x) # x is hidden_states
        savex = x+actualx
        x = self.ffnorm(savex)
        x = self.mLP(x)
        x = savex+x

        embed = x
        self.weight = weights
        # feature fusion
        num,inx = self.maws(weight,cont)
        for i in range(inx.shape[0]):
            #tokens[i].extend(embed[i,inx[i,:NUM_TOKEN]])
            self.toextend = [i,embed[i,inx[i,:NUM_TOKEN]]]

        return embed

class LastLayer(torch.nn.Module):
    def __init__(self,layer,size,tokens):
        super(LastLayer,self).__init__()
        self.layer = layer
        super(Block, self).__init__()
        self.attentnorm = torch.nn.LayerNorm(HIDDEN_SIZE,eps=1e-6)
        self.ffnorm = torch.nn.LayerNorm(HIDDEN_SIZE,eps=1e-6)
        self.mLP = MLP()
        self.attn = newAttention(HIDDEN_SIZE) # hidden_size = dim param
        
        self.encoder = torch.nn.LayerNorm(HIDDEN_SIZE,eps=1e-6) 

        return

    def forward(self, embeds):
        tokens=[torch.stack(token) for token in tokens]
        tokens = torch.stack(tokens).squeeze(1)
        concat = torch.cat((embeds[:,0].unsqueeze(1), tokens), dim=1)

        x = embeds
        actualx = x
        x = self.attentnorm(x)
        x,weights,cont = self.attn(x)
        savex = x+actualx
        x = self.ffnorm(savex)
        x = self.mLP(x)
        x = savex+x
        ff_states, ff_weights = x, weights
        # do last layer with selected 'important' tokens from MAWS
        encoded = self.encoder(ff_states)
        return encoded, weights

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
        model_conf: Optional[dict]=None  #"optimizer_name":'SGD'}
        # what is optim_kw?
    ):
        # training is done with gradient accumulation steps, so no extra code needed for optimizer
        self.feature_size = feature_size    

        # parent class initialization
        ImageClassifier.__init__(self, base_conf=base_conf, model_conf=model_conf)
    
    def setup_model(self):
        # for each backbone block, change to incorporate module
        hidden_size = self.backbone.embed_dim
        for i in range(len(self.backbone.blocks)):
            block = self.backbone.blocks[i]
            # each block takes embeddings!!
            self.backbone.blocks[i] = NewBlock(block, hidden_size)

        # edit last block
        #block = self.backbone.blocks[-1]
        #self.backbone.blocks[-1] = LastLayer(block,hidden_size)

        self.lastlayer = Block(hidden_size)

        self.encoder = torch.nn.LayerNorm(hidden_size,eps=1e-6) 
        
        # remove the ViT classification head
        self.backbone.head = torch.nn.Identity()

        # self.head = torch.nn.Sequential( # TODO:
        #     # torch.nn.BatchNorm1d(hidden_size * 3),
        #     torch.nn.LayerNorm(hidden_size * 3),
        #     torch.nn.Linear(hidden_size * 3, 1024),
        #     # torch.nn.BatchNorm1d(1024),
        #     torch.nn.LayerNorm(1024),
        #     torch.nn.ELU(inplace=True),
        #     torch.nn.Linear(1024, self.model_conf.num_classes),
        # )
        self.head = torch.nn.Linear(self.feature_size, self.num_classes)

    def forward(self, x):
        # x = x.transpose(-1,-2) # TODO: needed for sizing??
        breakpoint()
        x = self.backbone.forward_features(x)
        # get weights from each block
        weights = []
        tokens=[[] for i in range(x.shape[0])]
        for block in self.backbone.blocks: # TODO: does this work
            tokens[block.toextend[0]].extend(block.toextend[1])
            weights.append[block.weight]
            block.weight = None

        # do last layer here??
        embeds = x
        tokens=[torch.stack(token) for token in tokens]
        tokens = torch.stack(tokens).squeeze(1)
        concat = torch.cat((embeds[:,0].unsqueeze(1), tokens), dim=1)
        # do last layer with selected 'important' tokens from MAWS
        ff_states, ff_weights = self.lastlayer(concat)
        encoded = self.encoder(ff_states) # normalization

        logits = self.head(x[:,0])
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
