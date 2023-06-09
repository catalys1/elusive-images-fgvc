'''
MGE-CNN, from "Learning Granularity-Aware Convolutional Neural Network for Fine-Grained Visual Classification" 
(https://arxiv.org/abs/2103.02788).

Adapted from https://github.com/lbzhang/MGE-CNN
'''
from typing import Optional

import torch

from .base import ImageClassifier, get_backbone

__all__ = [
    'MGE-CNN',
]

################################################################################
# Internal Modules
################################################################################


################################################################################
# Lightning Module for SAC
################################################################################

