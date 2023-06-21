import argparse
import os
from pathlib import Path
import re

from omegaconf import OmegaConf


def make_config(dir, ckpt):
    dir = Path(dir)
    config = OmegaConf.load(dir.joinpath('config.yaml'))
    config.trainer.logger = False
    config.trainer.callbacks = None

    ckpt_dir = dir.joinpath('checkpoints')
    if Path(ckpt).is_file():
        ckpt_path = Path(ckpt)
    else:
        ckpt_path = ckpt_dir.joinpath(ckpt).with_suffix('.ckpt')

    if not ckpt_path.is_file():
        print(f'No matching checkpoints found in {str(ckpt_dir)}; skipping')
        return

    config.ckpt_path = str(ckpt_path)
    OmegaConf.save(config, dir.joinpath('pred_config.yaml'))


def make_prediction_configs(logdir, recurse=False, ckpt='last', exclude=None, force=False):
    logdir = Path(logdir)
    
    if recurse:
        exclude = exclude or []
        runs = []
        for root, dirs, files in os.walk(logdir):
            if 'config.yaml' in files:
                if force or ('preds.pth' not in files):
                    runs.append(Path(root))
            keep = []
            for dir in dirs:
                if not any(re.match(p, dir) for p in exclude):
                    keep.append(dir)
            dirs.clear()
            dirs.extend(keep)
        for run in runs:
            print(str(run))
        response = input('Do you wish to create prediction configs for these runs? [N|y] ')
        while True:
            if response.lower() in ('n', ''):
                print('Aborting')
                return
            elif response.lower() == 'y':
                break
            response = input('[N|y] ')
        for run in runs:
            make_config(run, ckpt)
    elif logdir.joinpath('config.yaml').is_file():
        make_config(logdir, ckpt)
    else:
        raise RuntimeError(f'Directory {str(logdir)} does not have a config.yaml')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', type=str, help='Path to root directory')
    parser.add_argument('-r', '--recurse', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-c', '--ckpt', type=str, default='epoch=49')
    parser.add_argument('-e', '--exclude', type=str, nargs='*', default=None)
    parser.add_argument('-f', '--force', action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args()

    make_prediction_configs(args.logdir, args.recurse, args.ckpt, args.exclude, args.force)