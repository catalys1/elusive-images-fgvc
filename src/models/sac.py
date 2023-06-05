'''
Self Assessment Classifier, from "Fine-Grained Visual Classification using Self Assessment Classifier" 
(https://arxiv.org/abs/2205.10529).

Adapted from https://github.com/aioz-ai/SAC
'''
from typing import Optional

import torch

from .base import ImageClassifier, get_backbone

__all__ = [
    'SAC',
]

################################################################################
# Internal Modules
################################################################################

class Classifier(torch.nn.Module):
    """
    Backbone Top-K ViT Classifier for SAC.
    """

################################################################################
# Lightning Module for SAC
################################################################################

class SAC(ImageClassifier):
    """
    Self Assessment Classifier
    """

    