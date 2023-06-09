'''Custom resolvers that can be used in configuration files through OmegaConf.
'''
from omegaconf import OmegaConf


# resolver that multiplies two values
OmegaConf.register_new_resolver("mul", lambda x, y: x*y)


def register(cache=False):
    '''Decorator helper function for registering a custom resolver.
    '''
    def decorator_register(func):
        OmegaConf.register_new_resolver(func.__name__, func, use_cache=cache)
        return func
    return decorator_register


@register(cache=False)
def path_seg(path, seg_idx=-1):
    '''Given a path made up of segments separated by "/", return the segment at seg_idx.
    '''
    segments = str(path).split('/')
    return segments[seg_idx]


@register(cache=False)
def linear_scale_factor(bs, base_bs, nodes=1, gpus_per_node=1):
    '''Compute a linear scaling factor for the learning rate based on the ratio of the batch size to
    a base batch size. Batch size is given in terms of a single GPU, so scaling needs to take into
    consideration the total number of distributed processes.
    '''
    return (bs / base_bs) * nodes * gpus_per_node

@register(cache=False)
def num_labels(datamod_path):
    '''Given a datamodule class_path, returns the number of labels associated with the
    dataset for classification.
    '''
    from src import data
    dm = datamod_path.rsplit('.', 1)[1]
    return getattr(data, dm).num_classes
