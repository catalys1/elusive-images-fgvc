import os
import re


def next_run_path(root):
    run = 0
    if os.path.exists(root):
        runs = [x for x in os.listdir(root) if re.match(r'run-\d+', x)]
        if len(runs) > 0:
            runs = sorted(int(x.split('-')[-1]) for x in runs)
            run = runs[-1] + 1
    path = os.path.join(root, f'run-{run}')
    return path
