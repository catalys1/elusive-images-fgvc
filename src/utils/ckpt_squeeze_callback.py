import os

import pytorch_lightning as pl
import torch


class CheckpointSqueezeCallback(pl.callbacks.Callback):
    def __init__(self, policy='all'):
        super().__init__()
        assert policy in ('all', 'last', 'best')
        self.policy = policy

    def on_train_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            if self.policy == 'all':
                dir = trainer.checkpoint_callback.dirpath
                files = [os.path.join(dir, x) for x in os.listdir(dir) if x.endswith('.ckpt')]
            elif self.policy == 'last':
                files = [trainer.checkpoint_callback.last_model_path]
            elif self.policy == 'best':
                files = [trainer.checkpoint_callback.best_model_path]
            files = [x for x in files if os.path.exists(x)]
            if len(files) == 0:
                print(f'Attempted to squeeze checkpoints based on policy {self.policy}, '
                       'but no matching checkpoint were found')
            for f in files:
                print(f'Squeezing checkpoint {f}')
                ckpt = torch.load(f, map_location='cpu')
                keep = ['epoch', 'global_step', 'pytorch-lightning_version', 'state_dict']
                ckpt = {k: ckpt[k] for k in keep}
                torch.save(ckpt, f)
