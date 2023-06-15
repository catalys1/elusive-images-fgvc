"""Utilities for helping to manage experiment logs and files.
"""
from pathlib import Path
import subprocess
import sys

sys.path.append('.')
from logs import get_wandb_run_from_id


def process_run_ids(run_ids):
    ids = []
    for rid in run_ids:
        if '-' in rid:
            s, e = rid.strip().split('-')
            ids.extend(list(range(int(s), int(e)+1)))
        else:
            ids.append(int(rid))
    ids = sorted(set(ids))
    return ids


def purge_runs(runs_path, run_ids, dryrun=True):
    '''Delete logging information for a given set of runs without removing configs or job launch
    scripts.
    '''
    # wandb_runs = list(Path(wandb_root).glob('offline-run-*'))
    if dryrun:
        print('Dryrun: the following files would be removed')
    run_ids = process_run_ids(run_ids)
    for n in run_ids:
        root = Path(f'{runs_path}/run-{n}')
        subdirs = ['checkpoints', 'slurm', 'job']
        for sd in subdirs:
            p = root.joinpath(sd)
            if p.exists():
                for x in p.iterdir():
                    if dryrun:
                        print(str(x))
                    else:
                        x.unlink()
                        print(f'{str(x)} ... removed')
                if not dryrun and sd == 'ckpt':
                    p.rmdir()
        wandb = list(root.joinpath('wandb').iterdir())
        for wf in wandb:
            if dryrun:
                if not wf.is_symlink():
                    print(str(wf))
            else:
                subprocess.run(['rm', '-r', str(wf)])
                print(f'{str(wf)} ... removed')


def relaunch_runs(runs_path, run_ids):
    purge_runs(runs_path, run_ids, dryrun=True)
    while True:
        response = input('Would you like to proceed? [N|y]: ')
        if response.lower() in ('n', ''):
            print('Aborting')
            break
        elif response.lower() == 'y':
            import jobs
            purge_runs(runs_path, run_ids, dryrun=False)
            run_ids = process_run_ids(run_ids)
            for n in run_ids:
                root = Path(f'{runs_path}/run-{n}')
                script = root.joinpath('job_submit.sh')
                jobs.launch_slurm_jobs([script])
            break


def cancel_jobs(start_jobid=None, end_jobid=None):
    output = subprocess.run(['squeue', '--me', '-o', '%i'], capture_output=True)
    ids = output.stdout.decode('ascii').strip().split('\n')[1:]
    ids.sort(key=lambda x: int(x))

    if start_jobid is None:
        start_jobid = ids[0]
    if end_jobid is None:
        end_jobid = ids[-1]
    start_jobid = int(start_jobid)
    end_jobid = int(end_jobid)

    ids = [x for x in ids if start_jobid <= int(x) <= end_jobid]

    print('The following jobs will be canceled:')
    print('\n'.join(f'  {x}' for x in ids))
    while True:
        response = input('Proceed? [N/y] ')
        if response.lower() in ('n', ''):
            print('Aborting')
            break
        elif response.lower() == 'y':
            subprocess.run(['scancel'] + ids)
            break


def sync_wandb_offline_runs(wandb_root='./wandb', runs_path=None, run_ids=None):
    '''Uses `wandb sync` to sync offline runs to the wandb server.
    
    Args:
        wandb_root: path to root wandb folder, where individual runs are stored.
        runs_path: (optional) path to root folder where run logs are stored, used in conjuction with `run_ids`
            to specify which runs to synchronize.
        run_ids: (optional) a list of integer run numbers to sync from. If runs_path and run_ids are specified,
            they will be used to decide which runs to sync. Otherwise, all unsynced runs under `wandb_root` will
            be synced.
    '''
    wandb_runs = list(Path(wandb_root).glob('offline-run-*'))
    to_sync = []
    if runs_path and run_ids:
        run_ids = process_run_ids(run_ids)
        for n in run_ids:
            run = Path(f'{runs_path}/run-{n}')
            wandb_id = run.joinpath('wandb_id').open().read().strip()
            wandb = get_wandb_run_from_id(wandb_runs, wandb_id)
            to_sync.append(str(wandb))
    else:
        for wandb in wandb_runs:
            wandb_id = wandb.name.rsplit('-', 1)[1]
            if wandb.joinpath(f'run-{wandb_id}.wandb.synced').exists():
                continue  # skip runs that have already been synced
            to_sync.append(str(wandb))
    print(f'Syncing {len(to_sync)} wandb runs')
    subprocess.run(['wandb', 'sync', '--no-include-synced', '--mark-synced'] + to_sync)
            

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    def t_or_f(arg):
        ua = str(arg).upper()
        if 'TRUE'.startswith(ua):
            return True
        elif 'FALSE'.startswith(ua):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    p = subparsers.add_parser('purge')
    p.add_argument('-d', '--dryrun', metavar="true/false", type=t_or_f, default=True)
    p.add_argument('-p', '--runs_path', type=str, default='./logs')
    p.add_argument('-i', '--run_ids', type=str, nargs='+', required=True)
    p.set_defaults(func=purge_runs)

    p = subparsers.add_parser('relaunch', aliases=['rl'])
    p.add_argument('-p', '--runs_path', type=str, default='./logs')
    p.add_argument('-i', '--run_ids', type=str, nargs='+', required=True)
    p.set_defaults(func=relaunch_runs)

    p = subparsers.add_parser('sync')
    p.add_argument('--wandb_root', type=str, default='logs/wandb')
    p.add_argument('--runs_path', type=str, default=None)
    p.add_argument('--run_ids', type=int, nargs='*', default=None)
    p.set_defaults(func=sync_wandb_offline_runs)

    p = subparsers.add_parser('cancel')
    p.add_argument('-s', '--start_jobid', type=int, default=None)
    p.add_argument('-e', '--end_jobid', type=int, default=None)
    p.set_defaults(func=cancel_jobs)

    args = parser.parse_args()
    args = args.__dict__
    func = args.pop('func')
    func(**args)
