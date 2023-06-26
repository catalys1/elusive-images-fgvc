'''Utilities for analyzing predictions (classifier outputs).
'''
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
from scipy.sparse.csgraph import connected_components
import torch
from torch import Tensor
from tqdm.auto import tqdm


def get_prediction_data(
    dataset: str,
    which: Optional[Union[int, Iterable]]=None,
    root: Union[str, Path]='../../logs',
    name: str='preds.pth',
    as_logits: bool=False,
    progress: bool=True,
) -> Tensor | Tuple[Tensor, Tensor]:
    '''Load and concatenate predictions from files.
    
    Args:
        dataset (str): name of the dataset, e.g. "cub".
        which (int | ints | None): the run number(s) to load from. If None, loads from all runs
            in the directory.
        root (str | path): root log directory. Should have a sub directory named as `dataset`.
        name (str): name of the file that contains the predictions.
        as_logits (bool): whether to return the full set of logits. Default is to return
            logits.argmax(-1), which is the class with the highest prediction for each image.
        progress (bool): whether to show a progress bar when loading the predictions.
    
    Returns:
        preds: tensor of shape (K, N) or (K, N, C) containing predictions or logits for each image
            in the dataset.
        labels: tensor of shape (N,) containing the ground-truth label for each image.    
    '''
    root = Path(root).joinpath(dataset)
    unpack = False
    if which is None:
        which = sorted([int(x.name.rsplit('-')[-1]) for x in root.glob('run-*')])
    elif isinstance(which, int):
        which = [which]
        unpack = True
    
    all_preds = []
    if progress:
        prog = tqdm(which, f'{dataset} ({name})')
    else:
        prog = which
    for i in prog:
        d = root.joinpath(f'run-{i}')
        data = torch.load(d.joinpath(name))
        preds = data['logits']
        if not as_logits:
            preds = preds.argmax(-1)
        all_preds.append(preds)
        labels = data['labels']
    preds = torch.stack(all_preds, 0)
    
    if unpack:
        preds = preds.squeeze()
    
    return preds, labels


def correct_mask(preds: Tensor, labels: Tensor, is_logits: bool=False) -> Tensor:
    '''Return binary mask indicating which values in preds are correct (match the corresponding
    value in labels).

    Args:
        preds (Tensor): predictions of shape (..., N) OR logits of shape (..., N, C) if
            is_logits=True.
        labels (Tensor): ground-truth labels of shape (N, ).
        is_logits (bool): if True, preds should contain logits (class scores) for each class.

    Returns:
        Boolean tensor of shape (..., N).
    '''
    if is_logits:
        preds = preds.argmax(-1)
    return preds.eq(labels)


def accuracy(preds: Tensor, labels: Tensor, is_logits: bool=False) -> Tensor:
    '''Return average accuracy over all predictions. The accuracy is given by

        sum(preds == labels) / len(labels)

    Args:
        preds (Tensor): predictions of shape (..., N) OR logits of shape (..., N, C) if
            is_logits=True.
        labels (Tensor): ground-truth labels of shape (N, ).
        is_logits (bool): if True, preds should contain logits (class scores) for each class.

    Returns:
        Tensor of shape (...) or (1, ) containing the average accuracy over all predictions.
    '''
    correct = correct_mask(preds, labels, is_logits)
    return correct.float().mean(-1)


def class_accuracy(
    preds: Tensor,
    labels: Tensor,
    is_logits: bool=False,
    num_class: Optional[int]=None,
) -> Tensor:
    '''Return average accuracy over all predictions for each class seperately.

    Args:
        preds (Tensor): predictions of shape (..., N) OR logits of shape (..., N, C) if
            is_logits=True; `...` indicates an optional first dimension.
        labels (Tensor): ground-truth labels of shape (N, ).
        is_logits (bool): if True, preds should contain logits (class scores) for each class.
        num_class (int | None): if given, defines the number of classes. Otherwise, the number of
            classes is assumed to be max(labels) + 1.

    Returns:
        Tensor of shape (C, ) containing the average accuracy for each class c = [0, 1, ..., C-1].
    '''
    if num_class is None:
        num_class = labels.max() + 1
    labels = labels.long()
    correct = correct_mask(preds, labels, is_logits)
    if correct.ndim == 2:
        index = (torch.arange(correct.shape[0])[:, None], labels[None])
        shape = (correct.shape[0], num_class)
    else:
        index = (labels,)
        shape = num_class
    class_correct = torch.zeros(shape).index_put_(index, correct.float(), accumulate=True)
    class_total = torch.zeros(shape).index_put_(index, torch.ones(1), accumulate=True)
    return class_correct / class_total.clamp_min_(1)


def prediction_overlap(preds: Tensor, labels: Tensor, return_img_count: bool=False) -> Tensor:
    '''Return prediction overlap: the number of images correctly predicted by k models, for
    k = [0, 1, ..., K].

    Args:
        preds (Tensor): K class predictions for each image with shape (K, N).
        labels (Tensor): ground-truth labels with shape (N, ).
    
    Returns:
        groups (Tensor): shape (K + 1, ), containing the number of images which were correctly predicted
            by a subset of k models, with k = [0, 1, ..., K].
        img_count (Tensor): returned if `return_img_count` is True; shape (N, ), contains for each image
            the number of correct predictions across the K models.

    '''
    img_count = correct_mask(preds, labels).sum(0)
    groups = img_count.bincount(minlength=preds.shape[0] + 1)
    if return_img_count:
        return groups, img_count
    else:
        return groups


def confusion_matrix(
    preds: Tensor,
    labels: Tensor,
    num_class: Optional[int]=None,
    class_idx: Optional[List[int]]=None,
    normalize: bool=False,
) -> Tensor:
    '''Return a confusion matrix for the given classes.

    The confusion matrix has shape (C, C), where C is the number of classes. The entry at (i, j) in the
    matrix gives the number (or proportion) of images in class `i` that were predicted as class `j`.

    Args:
        preds (Tensor): predictions for each image with shape (N, ).
        labels (Tensor): ground-truth labels with shape (N, ).
        num_class (int | None): if given, defines the number of classes. Otherwise, the number of
            classes is assumed to be max(labels) + 1.
        class_idx (list of int | None): if given, indicates a subset of classes which will be used to compute
            the confusion matrix. Otherwise, all classes are used.
        normalize (bool): if True, normalize each row of the matrix to sum to 1, so that each entry in row `i`
            gives the proportion of images from class `i` predicted as class `j`. If False, no normalization
            is applied and the matrix contains raw counts.
            When class_idx is given, each row is still normalized by the total number of images with that
            ground-truth label, so that the row sum may be less than one (when predictions fall outside the
            the given set of classes).

    Returns:
        confusion matrix (Tensor): confusion matrix with shape (C, C).
    '''
    if num_class is None:
        num_class = labels.max() + 1
    
    labels = labels.long()
    
    ncls = num_class
    if class_idx is not None:
        ncls = len(class_idx)
        class_idx = torch.tensor(class_idx)
        lookup = torch.empty(num_class, dtype=torch.int64).fill_(num_class)
        lookup.index_put_((class_idx,), torch.arange(ncls))
        label_mask = labels[:, None].eq(class_idx).any(-1)
        pred_mask = preds[:, None].eq(class_idx).any(-1)
        mask = label_mask & pred_mask
        rows = lookup[preds[mask]]
        cols = lookup[labels[mask]]
    else:
        rows = labels
        cols = preds

    conf = torch.zeros(ncls, ncls)
    conf.index_put_((rows, cols), torch.ones(1), accumulate=True)

    if normalize:
        if class_idx is None:
            conf = torch.nn.functional.normalize(conf, p=1, dim=-1)
        else:
            totals = labels.bincount()[class_idx].view(-1, 1)
            conf.div_(totals)

    return conf


def similar_class_groups(conf_mat: Tensor, threshold: Union[int, float]) -> List[Tensor]:
    '''Return groups of classes that are mutually confusing. A group of classes is mutually confusing
    if, for each class in the group, images from that class are frequently predicted as belonging to one
    or more other classes in the group. We find groups of mutually confusing classes by treating the
    confusion matrix as a graph and finding connected components.

    Args:
        conf_mat (Tensor): confusion matrix of shape (C, C).
        threshold (int | float): threshold on the number of predictions that will be considered an edge
            in the graph. If entry (i, j) in the confusion matrix has a value >= threshold, we create
            an edge between class i and class j.

    Returns:
        groups (List of Tensors): each tensor in the list contains indices of classes that are mutually
            confusing; that is, they belong to the same connected component in the thresholded confusion
            matrix. Only contains groups with more than a single class.
    '''
    edges = conf_mat.ge(threshold).long().numpy()
    components = torch.from_numpy(connected_components(edges, return_labels=True)[1])
    sizes = components.bincount()
    keeping = sizes.gt(1).nonzero().ravel()
    
    groups = []
    for k in keeping:
        g = components.eq(k).nonzero().ravel()
        groups.append(g)
    
    return groups

def topk(preds: Tensor, labels:Tensor, is_logits:bool=True, k: int=1) -> float:
    '''Return the top-k accuracy for the given predictions. 
    For now, needs logits with extra C dim, and assumes you give it this.

    Args:
        preds (Tensor): predictions of shape (..., N) OR logits of shape (..., N, C) if
            is_logits=True. C = number of classes.
        labels (Tensor): ground-truth labels of shape (N, ). N is number of images.
        k (int): number of top predictions to consider.
        is_logits (bool): if True, preds should contain logits (class scores) for each class.

    Returns:
        The accuracy at the given k.
    '''

    # preds is (K,N,C), K is number of runs, N is number of images, C is number of classes.
    #if not is_logits:
        # convert (later)

    # vals, indices
    top, topk_preds = preds.topk(k, -1) # topk with last dimension. topk_preds is [K,N,k]
    correct = topk_preds.eq(labels[:, None]).any(-1) # add dimension to labels and compare, broadcasting
    return correct.float().mean()


def getprobs(preds:Tensor) :
    '''Returns a prediction tensor where the last dimension is normalized to probabilities for each class (from logits)
     '''
    norm = torch.nn.functional.softmax(preds,dim=-1) # softmax logits to probabilities
    return norm


def log2ent(logits:Tensor) -> Tensor:
    '''Converts logits to have the last dimension be the entropy value of the predictions' probabilities.

    Args:
        logits: (...,N,C) Tensor predictions

    Returns:
        ents: (..., N) with entropy values for each image's predictions.
    '''
    probs = getprobs(logits)
    element_entr = torch.special.entr(probs)
    ents = torch.sum(element_entr, dim=-1)
    return ents

def class_entropy(preds:Tensor,labels:Tensor) -> Tensor:
    '''Return the average entropy and stdev of the entropies of the predictions for 
    each class in labels.
    (unsure if this is even useful but it was helpful to implement to figure out how the data is laid out)

    Args:
        preds: (...,N,C) Tensor predictions
        labels: (N,) Tensor ground truth labels
    Return: 
        (C,) Tensor entropy of predictions for each class'''

    # Not sure if this would be actually helpful or not. 
    # Can also modify to do for each run as well with .repeat()
    entropies = log2ent(preds)
    class_entropies = torch.zeros(preds.shape[-1])
    for i in range(preds.shape[-1]): # for each class
        class_ents = entropies[:,...,(labels==i)]
        # entropies[:,...,(labels==i)]
        # torch.masked_select(entropies,labels==i)
        # or get mean for each B
        class_entropies[i] = torch.mean(class_ents) #,torch.std(class_ents))

    return class_entropies

def correctk(preds: Tensor, labels: Tensor, is_logits: bool) -> int:
    '''Find the k value where all images are correct.
    
    Args:
        preds (Tensor): predictions of shape (..., N) OR logits of shape (..., N, C) if
            is_logits=True; `...` indicates an optional first dimension.
        labels (Tensor): ground-truth labels of shape (N, ).
        is_logits (bool): if True, preds should contain logits (class scores) for each class.
    '''
    accuracy = 0.0
    k=1

    while accuracy < 1.0:
        accuracy = topk(preds, labels, k=k, is_logits=is_logits)
        k += 1

    return k
