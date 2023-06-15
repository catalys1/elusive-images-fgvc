from .base import ImageClassifier, get_backbone
import torch
from typing import Optional
import timm
import copy # TODO: needed for deepcopy
# adapted from https://github.com/chou141253/FGVC-PIM/

FPN_SIZE = 512 # feature size - standardize

class WeakSelect(torch.nn.Module) :
    def __init__(x,num_classes,num_select):
        super(WeakSelect,self).__init__()
        # dict has x from backbone
        # numclasses, numselect
        self.num_select = num_select
        self.fpn_size = FPN_SIZE
    
    def forward(self,x, logits=None):
        # x dimensions should be [B, HxW, C] ([B, S, C]) or [B, C, H, W]
        # logits={}
        selections={}
        for name in x:
            if len(x[name].size()) == 4:
                B, C, H, W = x[name].size()
                x[name] = x[name].view(B, C, H*W).permute(0, 2, 1).contiguous()
            C = x[name].size(-1)
            
            probs = torch.softmax(logits[name], dim=-1)
            selections[name] = []
            preds_1 = []
            preds_0 = []
            num_select = self.num_select[name]
            for bi in range(logits[name].size(0)):
                max_ids, _ = torch.max(probs[bi], dim=-1)
                confs, ranks = torch.sort(max_ids, descending=True)
                sf = x[name][bi][ranks[:num_select]]
                nf = x[name][bi][ranks[num_select:]]  # calculate
                selections[name].append(sf) # [num_selected, C]
                preds_1.append(logits[name][bi][ranks[:num_select]])
                preds_0.append(logits[name][bi][ranks[num_select:]])
            
            selections[name] = torch.stack(selections[name])
            preds_1 = torch.stack(preds_1)
            preds_0 = torch.stack(preds_0)

            logits["select_"+name] = preds_1
            logits["drop_"+name] = preds_0

        return selections # TODO: fix dimensions


# features pyramid network
class FPN(torch.nn.Module):
    def __init__(inputs):
        super(FPN,self).__init__()
        # using convolutional model for ViT, if ResNet use linear.
        self.inputs = inputs # x's from backbone
        inp_names = [name for name in inputs]
        self.fpn_size = FPN_SIZE
        # LOOK AT PROCESS FROM FFVT LAYERS!!
        for i,node_name in enumerate(inputs):
            mod = torch.nn.Sequential(
                        torch.nn.Conv2d(inputs[node_name].size(1), inputs[node_name].size(1), 1),
                        torch.nn.ReLU(),
                        torch.nn.Conv2d(inputs[node_name].size(1), fpn_size, 1))
            
            self.add_module("Proj_"+node_name,mod)
            
            if i!=0:
                # customized upsample module
                assert len(inputs[node_name].size()) == 3 # B, S, C
                in_dim = inputs[node_name].size(1)
                out_dim = inputs[inp_names[i-1]].size(1)
                if in_dim != out_dim:
                    m = torch.nn.Conv1d(in_dim, out_dim, 1) # for spatial domain
                else:
                    m = torch.nn.Identity()
                
                self.add_module("Up_"+node_name, m) # TODO: module needed/streamline?

    def forward(self,x):
        hs = []
        for i, name in enumerate(x):
            x[name] = getattr(self, "Proj_"+name)(x[name])
            hs.append(name)

        for i in range(len(hs)-1, 0, -1):
            x1_name = hs[i]
            x0_name = hs[i-1]
            # add upsample
            x[x0_name] = getattr(self, "Up_"+x1_name)(x[x1_name]) + x[x0_name]
        return x


class Combiner(torch.nn.Module):
    def __init__(total_num_selects, num_classes, inputs):
        # build one layer structure (with adaptive module)

        # proj_size is fpn_size
        self.proj_size = FPN_SIZE
        num_joints = total_num_selects // 32

        self.param_pool0 = torch.nn.Linear(total_num_selects, num_joints)

        A = torch.eye(num_joints)/100 + 1/100
        self.adj1 = torch.nn.Parameter(copy.deepcopy(A))
        self.conv1 = torch.nn.Conv1d(self.proj_size, self.proj_size, 1)
        self.batch_norm1 = torch.nn.BatchNorm1d(self.proj_size)
        
        self.conv_q1 = torch.nn.Conv1d(self.proj_size, self.proj_size//4, 1)
        self.conv_k1 = torch.nn.Conv1d(self.proj_size, self.proj_size//4, 1)
        self.alpha1 = torch.nn.Parameter(torch.zeros(1))

        ### merge information
        self.param_pool1 = torch.nn.Linear(num_joints, 1)
        
        #### class predict
        self.dropout = torch.nn.Dropout(p=0.1)
        self.classifier = torch.nn.Linear(self.proj_size, num_classes)

        self.tanh = torch.nn.Tanh()

    def forward(self,x):
        # TODO: standardize naming conventions?
        hs = []
        for name in x:
            hs.append(hs.append(x[name]))
        
        hs = torch.cat(hs, dim=1).transpose(1, 2).contiguous()
        hs = self.param_pool0(hs)
        ### adaptive adjacency
        q1 = self.conv_q1(hs).mean(1)
        k1 = self.conv_k1(hs).mean(1)
        A1 = self.tanh(q1.unsqueeze(-1) - k1.unsqueeze(1))
        A1 = self.adj1 + A1 * self.alpha1
        ### graph convolution
        hs = self.conv1(hs)
        hs = torch.matmul(hs, A1)
        hs = self.batch_norm1(hs)
        ### predict
        hs = self.param_pool1(hs)
        hs = self.dropout(hs)
        hs = hs.flatten(1)
        hs = self.classifier(hs)

        return hs

class PIM(ImageClassifier):
    def __init__(
        self,
        feature_size: int=512,
        base_conf: Optional[dict]=None,
        model_conf: Optional[dict]=None  #"optimizer_name":'SGD'}
    ):
        self.feature_size = feature_size    
        self.fpn_size = FPN_SIZE
        # parent class initialization
        ImageClassifier.__init__(self, base_conf=base_conf, model_conf=model_conf)
    
    def setup_model(self):
        # TODO: get backbone outputs/return nodes from backbone here (check if needed)

        # default return_nodes and num_selects for ViT backbone in builder.py
        return_nodes = {
            'blocks.8': 'layer1',
            'blocks.9': 'layer2',
            'blocks.10': 'layer3',
            'blocks.11': 'layer4',
        }

        # add feature_extractor TODO: ask if this is needed!
        self.backbone = torchvision.models.feature_extraction.create_feature_extractor(self.backbone, return_nodes=return_nodes)
        rand_in = torch.randn(1, 3, 224, 224)
        x = self.backbone(rand_in) # TODO: check if backbone is instantiated before setup_model so this works

        num_selects = {
            'layer1':32,
            'layer2':32,
            'layer3':32,
            'layer4':32
        }

        self.fpn = FPN(x, FPN_SIZE)
        self.build_fpn_classifier(x, FPN_SIZE, self.num_classes)

        # classifier is built into weak selector
        # self.classifier = torch.nn.Linear(self.feature_size,self.num_classes)
        self.selector = WeakSelect(x,self.num_classes,num_selects)

        gcn_inputs, gcn_proj_size = None, None
        total_num_selects = sum([num_selects[name] for name in num_selects]) # sum
        self.combiner = Combiner(total_num_selects, self.num_classes, gcn_inputs)

    def build_fpn_classifier(self, inputs, fpn_size, num_classes):
        for name in inputs:
            m = torch.nn.Sequential(
                    torch.nn.Conv1d(fpn_size, fpn_size, 1),
                    torch.nn.BatchNorm1d(fpn_size),
                    torch.nn.ReLU(),
                    torch.nn.Conv1d(fpn_size, num_classes, 1))
            self.add_module("fpn_classifier_"+name, m)
            # TODO: do modules work with this? is this a torch Module?

    def forward(self, x):
        logits = {}
        x = self.backbone(x)
        x = self.fpn(x)

        # predict for each feature point
        for name in x:
            if len(x[name].size()) == 4:
                B, C, H, W = x[name].size()
                logit = x[name].view(B, C, H*W)
            elif len(x[name].size()) == 3:
                logit = x[name].transpose(1, 2).contiguous()
            logits[name] = getattr(self, "fpn_classifier_"+name)(logit)
            logits[name] = logits[name].transpose(1, 2).contiguous()

        selected = self.selector(x,logits)
        combined = self.combiner(selected)
        logits['comb_outs'] = comb_outs # keep comb_outs? check what they are used for
        
        # this module returns "layer#"s, "preds_0" and "preds_1" from Selector, and "comb_outs" from Combiner
        # for eval, these were originally put through a separate loss process in training step.
        # TODO: lines 141-184 in main.py
        return logits

