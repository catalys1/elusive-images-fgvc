from itertools import product
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
import utils as xu


cluster = dict(
    cpus_per_task = 8,
    nodes = 1,
    gpus = 1,
    mem = 24,
    qos = 'cs',
    time = '',
)

datasets = [
    'cub',
    'aircraft',
    'cars',
    'nabirds',
    'fungi',
]

models = [
    'resnet50',
    'vit',
    'pmg',
    'simtrans',
    'wsdan',
    'ielt',
]

data_epoch_times = {
    'cub': 1,
    'aircraft': 1,
    'cars': 1,
    'nabirds': 4,
    'fungi': 5,
}

model_epoch_times = {
    'resnet50': 1,
    'vit': 1.25,
    'pmg': 2,
    'simtrans': 3,
    'wsdan': 2,
    'ielt': 1.5,
}

seeds = range(5)

runs = []
for dataset, model, seed in product(datasets, models, seeds):
    clu = {k: v for k, v in cluster.items()}
    trainer_conf = xu.find_config('trainer')
    data_conf = xu.find_config(dataset)
    model_conf = xu.find_config(model)
    conf = xu.combine_from_files(trainer_conf, data_conf, model_conf)

    # set random seed
    conf.seed_everything = f'${{set_seed:{seed}}}'

    # scale learning rate if using larger batch size
    bs = conf.data.init_args.batch_size
    if bs > 16:
        conf.model.init_args.base_conf.base_lr *= (bs / 16)

    clu['time'] = int(conf.trainer.max_epochs * data_epoch_times[dataset] * model_epoch_times[model])
    if dataset in ('fungi', 'nabirds'):
        clu['mem'] = 48

    log_subdir = f'{dataset}'
    run = xu.setup_run_for_slurm(conf, clu, log_dir='logs', log_subdir=log_subdir, prefer='config')
    runs.append(run)

xu.launch_slurm_jobs(runs, 0.2)
