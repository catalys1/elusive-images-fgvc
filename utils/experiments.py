import json
from pathlib import Path
import re
import sys

import numpy as np
from omegaconf import OmegaConf

if __name__ == '__main__':
    sys.path.append('.')
    from configs import pselect, find
    from logs import *
else:
    from .configs import pselect, find
    from .logs import *


__all__ = [
    'RunData',
]


class RunData(object):
    def __init__(self, root):
        self.root = Path(root)
        self.wandb_root = self.root.joinpath('wandb')
        
        self.wandb_run_paths = sorted(self.wandb_root.glob('*run-*-*'))
        self.wandb_id = self.wandb_run_paths[-1].name.rsplit('-', 1)[-1]

        self.config = OmegaConf.load(self.root.joinpath('raw_run_config.yaml'))
        
        self._metrics = None

    @property
    def name(self):
        return self.root.name

    @property
    def run_id(self):
        return int(self.name.rsplit('-', 1)[1])

    @property
    def summary(self):
        return json.load(self.wandb_run_paths[-1].joinpath('files', 'wandb-summary.json').open())

    @property
    def metadata(self):
        return json.load(self.wandb_run_paths[-1].joinpath('files', 'wandb-metadata.json').open())

    @property
    def model_type(self):
        if not hasattr(self, '_model_type'):
            self._model_type = get_model_type(self)
        return self._model_type
    
    @property
    def metrics(self):
        if self._metrics is None:
            self._metrics = []
            for p in self.wandb_run_paths:
                wpath = p.joinpath(f'run-{self.wandb_id}.wandb')
                mets = extract_wandb_metrics(wpath)
                self._metrics.extend(mets)
        return self._metrics
    
    @property
    def grouped_metrics(self):
        metrics = self.metrics
        grouped = {k: [] for k in metrics[0]}
        for m in metrics:
            for k in m:
                if not np.isnan(m[k]):
                    grouped[k].append(m[k])
        return grouped

    @property
    def val_metrics(self):
        val_keys = sorted([k for k in self.metrics[0] if k.startswith('val/')])
        return [
            {k: x[k] for k in val_keys} for x in self.metrics
            if all(not np.isnan(x[k]) for k in val_keys)
        ]

    @property
    def max_val_metrics(self):
        if not hasattr(self, '_max_val_metrics'):
            metrics = self.val_metrics
            self._max_val_metrics = {k: max(x[k] for x in metrics) for k in metrics[0]}
        return self._max_val_metrics

    def get_metric(self, name):
        metrics = self.metrics
        return [x[name] for x in metrics if not np.isnan(x[name])]

    def clear_metrics(self):
        self._metrics = None
        if hasattr(self, '_max_val_metrics'):
            del self._max_val_metrics

    def confv(self, val):
        return pselect(self.config, val)

    def print_conf(self, key=None):
        conf = self.config
        if key is not None:
            conf = OmegaConf.select(conf, key)
        print(OmegaConf.to_yaml(conf))

    def __getitem__(self, val):
        return self.confv(val)

    def __repr__(self):
        s = f'<Run ({self.name})>'
        return s

    def __str__(self):
        return self.__repr__()


def get_model_type(run):
    name = run['model.init_args.*.name'][0][1]
    if name == 'facebook/vit-mae-base':
        return 'ViT'
    size = getif(run['model.init_args.*.hidden_size'])
    if size and size == 384:
        return 'RoBERTa-small'
    return 'RoBERTa-base'


def getif(x):
    if len(x):
        return x[0][1]
    return None


def get_run_list(run_dir, id_range=None, return_type='path'):
    def get_id(path: Path):
        return int(path.name[len('run-'):])

    run_paths = sorted(Path(run_dir).glob('run-*'), key=get_id)
    if id_range is None:
        id_range = range(0, get_id(run_paths[-1]) + 1)
    if isinstance(id_range, str):
        low, high = map(int, id_range.split('-'))
        id_range = range(low, high + 1)
    if return_type == 'rundata':
        runs = [RunData(r) for r in run_paths if get_id(r) in id_range]
    elif return_type == 'path':
        runs = [r for r in run_paths if get_id(r) in id_range]
    return runs


def compare(runs, keys=None, time=False):
    keys = keys or []
    vals = []
    for r in runs:
        if isinstance(r, Path):
            r = str(r)
        log = list(Path(r).joinpath('slurm').glob('*.out'))[0].open().read()
        config = OmegaConf.load(Path(r).joinpath('raw_run_config.yaml'))
        val_acc = [float(x) for x in re.findall('Val epoch \d+/\d+: .*val/acc = (0\.\d+)', log)]
        max_val = max(val_acc) if val_acc else '---'
        vs = [max_val]
        for k in keys:
            v = find(config, k, '---')
            if 'class_path' in k:
                v = v.rsplit('.', 1)[-1]
            vs.append(v)
        if time:
            t = re.search(r'Total training time (.*)', log)
            if t: t = t.group(1)
            else: t = '---'
            vs.append(t)
        vals.append(vs)
    
    if time:
        keys.append('Time')
    slens = [max([len(keys[i])] + [len(str(v[i + 1])) for v in vals]) + 2 for i in range(len(keys))]
    row = ('{{:<{}}}' * (len(keys) + 2))
    row = row.format(8, 9, *slens)
    print('-' * (sum(slens) + 17))
    print(row.format('Run', 'Val acc', *keys))
    print('-' * (sum(slens) + 17))
    for run, vs in zip(runs, vals):
        s = row.format(run.name, *[str(x) for x in vs])
        print(s)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=str)
    parser.add_argument('-k', '--keys', nargs='*', default=None)
    parser.add_argument('-r', '--range', type=str, default=None)
    parser.add_argument('--time', action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()
    runs = get_run_list(args.run_dir, args.range)
    compare(runs, args.keys, args.time)
