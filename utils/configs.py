'''Utilities for working with configuration files and omegaconf configs.'''
import os

import omegaconf
from omegaconf import OmegaConf


__all__ = [
    'pselect',
    'compare_configs',
    'get_config_diffs',
    'remove',
    'combine_from_files',
    'find_config',
]


def check_key(k, path_segs):
    path = '.'.join(path_segs)
    if not '*' in k:
        return k in path
    parts = k.split('*')
    i, j = 0, 0
    for part in parts:
        i = path.find(part, j)
        if i == -1:
            return False
        j = i + len(part)
    return True


def pselect(config, key):
    '''Recursively search for partial matches to `key` in `config`. `key` can contain the `*` character,
    which is a wild character that matches any contiguous string of characters, as in Unix glob patterns.
    '''
    matches = []
    path = []

    def get_val(conf, key):
        if OmegaConf.is_interpolation(conf, key) or OmegaConf.is_missing(conf, key):
            v = conf._get_node(key)._value()
        else:
            v = conf[key]
        return v

    def recurse(conf):
        if isinstance(conf, (dict, omegaconf.DictConfig)):
            _iter = conf
        else:
            _iter = range(len(conf))
        for k in _iter:
            path.append(str(k))
            if check_key(key, path):
                v = get_val(conf, k)
                matches.append(('.'.join(path), v))
            elif OmegaConf.is_interpolation(conf, k) or OmegaConf.is_missing(conf, k):
                pass
            elif isinstance(conf[k], (dict, omegaconf.DictConfig, list, omegaconf.ListConfig)):
                recurse(conf[k])
            path.pop()
    
    recurse(config)
    return matches


def find(config, key, *default):
    '''
    Args:
        config (config): OmegaConf configuration.
        key (str): key to search for; can contain '*', which will match any contiguous section.
        default (any): optional default value to return if key is not found. If omitted, a KeyError
            is raised if key cannot be found in config.
    '''
    if len(default) > 1:
        raise ValueError(f'Found {len(default)} additional arguments, but only accepts 1')
    result = pselect(config, key)
    if len(result) == 0:
        if len(default) == 0:
            raise KeyError(f'No matches found for key ({key})')
        else:
            return default[0]
    elif len(result) > 1:
        raise KeyError(f'Found multiple matches for key ({key}): ' + ', '.join(x[0] for x in result))
    return result[0][1]


def compare_configs(conf1, conf2, ignore=None, include=None):
    '''Compare two config files, and return the differences. Every leaf node in the two configs are
    compared, and the differences are returned as a list of (key, val1, val2).
    Differences include when a key exists in one config but not the other (in which case None is returned
    as the value for the config with the missing key), or when the value for the same key is different
    between the two configs.

    Args:
        conf1: first config (dict or OmegaConf)
        conf2: second config (dict or OmegaConf)
        ignore: a list of key patterns to ignore. Partial matches are treated as a match, and "*" can be
            used as a wild, as in GLOB patterns.
    '''
    path = []
    keys = []
    ignore = ignore or []
    include = include or []

    iterables = (dict, omegaconf.DictConfig, list, omegaconf.ListConfig)

    def check_ignore(path):
        v = False
        for k in ignore:
            if check_key(k, path):
                v = True
                break
        return v

    def recurse(conf):
        if isinstance(conf, (dict, omegaconf.DictConfig)):
            _iter = conf
        else:
            _iter = range(len(conf))
        for k in _iter:
            path.append(str(k))
            if check_ignore(path):
                path.pop()
                continue
            if OmegaConf.is_interpolation(conf, k) or OmegaConf.is_missing(conf, k):
                pass
            elif isinstance(conf[k], iterables):
                recurse(conf[k])
            else:
                keys.append('.'.join(path))
            path.pop()
            
    recurse(conf1)
    recurse(conf2)
    keys.extend(include)
    keys = set(keys)
    
    mismatch = []
    for k in keys:
        v1 = OmegaConf.select(conf1, k, default=None)
        v2 = OmegaConf.select(conf2, k, default=None)
        if isinstance(v1, iterables) or isinstance(v2, iterables):
            continue
        if v1 != v2:
            mismatch.append((k, v1, v2))
    return mismatch


def get_config_diffs(config_list, ignore_list=None, include_list=None):
    diffs = []
    c0 = config_list[0]
    for c in config_list[1:]:
        diffs.append(compare_configs(c0, c, ignore_list, include_list))
    return diffs


def remove(conf, path):
    if '.' in path:
        p, x = path.rsplit('.', 1)
        p = OmegaConf.select(conf, p)
        if p and x in p:
            delattr(p, x)
    else:
        if path in conf:
            delattr(conf, path)


def combine_from_files(*configs, **named_configs):
    '''Create a single config by loading and merging multiple config files. The named_configs
    get added as nodes using their name as the root key.
    '''
    configs = [OmegaConf.load(c) for c in configs]
    configs.extend([OmegaConf.create({k: OmegaConf.load(c)}) for k, c in named_configs.items()])
    config = OmegaConf.merge(*configs)
    return config


def find_config(name, root='src/configs'):
    if not name.endswith('.yaml'):
        name = f'{name}.yaml'
    
    path = None
    for r, dirs, files in os.walk(root):
        if name in files:
            path = os.path.join(r, name)
            break

    return path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('configs', nargs='+')
    parser.add_argument('-f', '--file', type=str, default='config.yaml')
    args = parser.parse_args()

    config = combine_from_files(*args.configs)

    OmegaConf.save(config, args.file)