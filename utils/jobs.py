'''Utilities for creating and launching jobs.'''
import os
import re
import subprocess
import time

import dotenv
from omegaconf import OmegaConf
import shortuuid


__all__ = [
    'cluster_defaults',
    'generate_id',
    'create_slurm_batch_file',
    'sync_cluster_and_config',
    'setup_run_for_slurm',
    'setup_run_for_local',
    'launch_slurm_jobs',
]


# load custom environment variables defined in `.env` at the root of the project
dotenv.load_dotenv(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/.env')

CONFIG_TEMPLATE = os.path.join(os.environ['WORKDIR'], 'mmnoise/configs/template.yaml')


def cluster_defaults():
    '''Return dictionary with default values for cluster parameters.'''
    return dict(
        nodes = 1,
        gpus = 1,
        mem = 40,
        cpus_per_task = 6,
        time = 360,
    )


def generate_id(length: int = 8) -> str:
    '''Generate a run id for wandb.'''
    # copied from wandb: https://github.com/wandb/wandb/blob/v0.13.0/wandb/util.py#L743
    run_gen = shortuuid.ShortUUID(alphabet=list("0123456789abcdefghijklmnopqrstuvwxyz"))
    return str(run_gen.random(length))


def next_run_path(root):
    '''Return the next numerically-sequential unused run path under `root`.'''
    run = 0
    if os.path.exists(root):
        runs = [x for x in os.listdir(root) if re.match(r'run-\d+', x)]
        if len(runs) > 0:
            runs = sorted(int(x.split('-')[-1]) for x in runs)
            run = runs[-1] + 1
    path = os.path.join(root, f'run-{run}')
    return path


def dedup_path(path):
    '''Remove duplicate forward slashes from a path string.'''
    return re.sub(r'//+', '/', path)


def create_slurm_batch_file(
    command,
    nodes = 1,
    gpus = 1,
    mem = 40,
    cpus_per_task = 6,
    time = 360,
    root_dir = os.environ['WORKDIR'],
    output_dir='logs',
    name = None,
    auto_resubmit = True,
    **kwargs,
):
    '''Generate the contents of a bash file that can be used to submit a job to slurm through sbatch.

    Args:
        command: string containing the python command to be executed, e.g. "python train.py fit --config=path"
        nodes: number of nodes
        gpus: number of gpus per node, world size will be nodes * gpus
        mem: amount of memory to allocate in GB
        cpus_per_task: number of cpus to allocate per task (or per gpu)
        time: allocated maximum time for the job, can be an int representing minutes, or a time string
            like 06:00:00 (hours:minutes:seconds)
        root_dir: path to root directory where command should be executed from
        output_dir: path to directory where logs will be created
    
    Returns:
        str, the full content of the batch file
    '''
    out_file = os.path.join(output_dir, 'slurm', 'slurm-%j.out')
    err_file = os.path.join(output_dir, 'slurm', 'slurm-%j.err')

    sbatch_opts = [
        f'--nodes={nodes}',
        f'--gres=gpu:{gpus}',
        f'--ntasks-per-node={gpus}',
        f'--cpus-per-task={cpus_per_task}',
        f'--mem={mem}G',
        f'--time={time}',
        f'--chdir={root_dir}',
        f'--output={out_file}',
        f'--error={err_file}',
    ]
    sbatch_opts += [f'--{k}={v}' for k, v in kwargs.items()]
    if name:
        sbatch_opts.append(f'--job-name={name}')
    if auto_resubmit:
        sbatch_opts.append(f'--signal=SIGUSR1@90')
        sbatch_opts.append(f'--requeue')
    sbatch_opts = '\n'.join(f'#SBATCH {x}' for x in sbatch_opts)
    sbatch = (
        '#!/bin/bash\n'
        f'{sbatch_opts}\n\n'
        'source ~/.bashrc\n'
        'source .env\n'
        f'conda activate $CONDA_ENV_NAME\n\n'
        f'srun -u {command}\n'
    )
    return sbatch


def sync_cluster_and_config(config, cluster_kw, conf_key, cluster_key, prefer='config'):
    '''Ensures that the config and cluster options are in sync for a given key. By default, priority
    is given to config values: cluster values will be overwritten when a config value exists. Otherwise,
    the cluster value is injected into the config. Passing prefer="cluster" will give priority to the
    cluster value instead.
    
    conf_key should be given as dotlist path, like arg.segment.key.
    '''
    if prefer not in ('config', 'cluster'):
        raise RuntimeError(f'`prefer` must be one of ("cluster", "config"), got "{prefer}"')
    conf_val = OmegaConf.select(config, conf_key, default='__missing__')
    cluster_val = cluster_kw.get(cluster_key, '__missing__')
    if conf_val == '__missing__' and cluster_val == '__missing__':
        raise RuntimeError('Must specify a value in either the config or the cluster dict')
    if conf_val == '__missing__':
        OmegaConf.update(config, conf_key, cluster_kw[cluster_key])
    elif cluster_val == '__missing__':
        cluster_kw[cluster_key] = conf_val
    elif prefer == 'config':
        cluster_kw[cluster_key] = conf_val   
    else:
        OmegaConf.update(config, conf_key, cluster_kw[cluster_key])


def _make_run_dir(log_dir, log_subdir=None, slurm=False):
    subdir = log_subdir or ''
    log_dir = log_dir or os.path.join(os.environ['WORKDIR'], 'logs', subdir)
    run_dir = dedup_path(next_run_path(os.path.join(log_dir, subdir)))
    os.makedirs(run_dir, exist_ok=False)  # should be a new run dir
    if slurm:
        os.makedirs(os.path.join(run_dir, 'slurm'))  # for slurm logs

    return run_dir


def _setup_wandb(run_dir, config):
    os.makedirs(os.path.join(run_dir, 'wandb'))  # for wandb logs
    wandb_id = generate_id()
    OmegaConf.update(config, 'trainer.logger.0.init_args.id', wandb_id)


def setup_run_for_slurm(config, slurm_kw=None, log_dir=None, log_subdir=None, prefer='config', with_wandb=True):
    '''Sets everything up for launching a run. Syncs config and sbatch parameters, creates the run directory
    and sets corresponding paths in the config, and saves the config and bash file for launching the job
    in the run directory.
    '''
    slurm_def = cluster_defaults()
    if slurm_kw:
        slurm_def.update(slurm_kw)
    slurm_kw = slurm_def

    # create the run log directory
    run_dir = _make_run_dir(log_dir, log_subdir, slurm=True)

    # update slurm and config with paths
    # log and checkpoint paths should be defined relative to trainer.default_root_dir using node interpolation
    _, rtype, rname = run_dir.rstrip('/').rsplit('/', 2)
    slurm_kw['output_dir'] = run_dir
    slurm_kw['root_dir'] = os.environ['WORKDIR']
    slurm_kw['name'] = f'{rname}:{rtype}'
    config.trainer.default_root_dir = run_dir

    # synchronize config and slurm args
    sync_cluster_and_config(config, slurm_kw, 'trainer.num_nodes', 'nodes', prefer=prefer)
    sync_cluster_and_config(config, slurm_kw, 'trainer.devices', 'gpus', prefer=prefer)
    sync_cluster_and_config(config, slurm_kw, 'data.init_args.num_workers', 'cpus_per_task', prefer=prefer)
    if slurm_kw['nodes'] > 1 or slurm_kw['gpus'] > 1:
        config.trainer.strategy = 'ddp'

    if with_wandb:
        # inject a wandb run id
        _setup_wandb(run_dir, config)
    else:
        # switch logger to default (tensorboard)
        config.trainer.logger = True

    # save files to run dir
    config_path = os.path.join(run_dir, 'raw_run_config.yaml')
    OmegaConf.save(config, config_path)
    command = f'python run.py fit --config={config_path}'
    slurm_file_content = create_slurm_batch_file(command, **slurm_kw)
    job_file = os.path.join(run_dir, 'job_submit.sh')
    with open(job_file, 'w') as f:
        f.write(slurm_file_content)

    # return path to slurm job file
    return job_file
    

def setup_run_for_local(config, log_dir, log_subdir=None, with_wandb=True):
    '''Sets up a run for local execution, without a job submission script.
    '''
    # create the run log directory
    run_dir = _make_run_dir(log_dir, log_subdir, slurm=False)

    # update config with paths
    # log and checkpoint paths should be defined relative to trainer.default_root_dir using
    # OmegaConf node interpolation
    config.trainer.default_root_dir = run_dir

    if config.trainer.num_nodes > 1 or config.trainer.devices > 1:
        config.trainer.strategy = 'ddp'

    # inject a wandb run id
    if with_wandb:
        _setup_wandb(run_dir, config)
    else:
        config.trainer.pop('logger')

    # save files to run dir
    config_path = os.path.join(run_dir, 'raw_run_config.yaml')
    OmegaConf.save(config, config_path)

    # return path to config for the run
    return config_path


def launch_slurm_jobs(job_file_list, sleep_time=None):
    '''Given a list of paths to slurm batch files, execute each using sbatch
    to submit the jobs to the slurm scheduler.
    '''
    for job_file in job_file_list:
        args = ['sbatch', job_file]
        subprocess.run(args)
        # possibly sleep so as not to overrun the scheduler
        if sleep_time is not None:
            time.sleep(sleep_time)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config', nargs='+', type=str)
    parser.add_argument('-d', '--log_dir', type=str, default='logs')
    parser.add_argument('-s', '--log_subdir', type=str, default='__tests')
    parser.add_argument('-w', '--with_wandb', type=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    import sys
    sys.path.append('.')
    import configs

    args.config = configs.combine_from_files(*args.config)

    setup_run_for_local(**args.__dict__)
