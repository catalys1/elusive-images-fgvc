import time

import pytorch_lightning as pl


class PrintProgressCallback(pl.callbacks.ProgressBar):
    def __init__(self, print_interval=20):
        super().__init__()
        self.is_enabled = True
        self.print_interval = print_interval
        self.rank_0 = True

    def enable(self):
        self.is_enabled = True
    
    def disable(self):
        self.is_enabled = False

    def print(self, *args, **kwargs):
        if self.rank_0:
            print(*args, flush=True, **kwargs)

    def on_train_start(self, trainer, pl_module):
        self.rank_0 = trainer.is_global_zero
        self.total_train_time = time.time()

    def on_train_epoch_start(self, *args, **kwargs):
        self.avg_time = 0
        self.steps = 0
        self.train_epoch_time = time.time()

    def on_train_batch_start(self, *args, **kwargs):
        self.start_time = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        elapse = time.time() - self.start_time
        self.avg_time += elapse
        self.steps += 1

        if batch_idx % self.print_interval == 1 or batch_idx == trainer.num_training_batches - 1:
            t = round(self.avg_time / self.steps, 4)
            tb = str(self.total_train_batches)
            b = str(batch_idx).zfill(len(tb))
            head = 'Train epoch {}/{} (batch {}/{}):  batch_time = {}  '.format(
                self.trainer.current_epoch + 1, self.trainer.max_epochs, b, tb, t
            )
            metrics = self.get_metrics(trainer)
            out = '   '.join(f'{k} = {v}' for k, v in metrics.items())
            self.print(head + out)

    def on_train_epoch_end(self, *args, **kwargs):
        t = round(time.time() - self.train_epoch_time, 4)
        self.print('Train epoch {} elapsed time: {}'.format(self.trainer.current_epoch + 1, t))

    def on_validation_epoch_start(self, *args, **kwargs):
        self.val_epoch_time = time.time()
        self.avg_time = 0
        self.steps = 0

    def on_validation_batch_start(self, *args, **kwargs):
        self.start_time = time.time()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        elapse = time.time() - self.start_time
        self.avg_time += elapse
        self.steps += 1

    def on_validation_epoch_end(self, trainer, pl_module):
        t = round(self.avg_time / self.steps, 4)
        head = 'Val epoch {}/{}:  batch_time = {}  '.format(
            self.trainer.current_epoch + 1, self.trainer.max_epochs, t
        )
        metrics = self.get_metrics(trainer, val=True)
        out = '   '.join(f'{k} = {v}' for k, v in metrics.items())
        self.print(head + out)

        t = round(time.time() - self.val_epoch_time, 4)
        self.print('Val epoch {} elapsed time: {}'.format(self.trainer.current_epoch + 1, t))

    def on_train_end(self, trainer, pl_module):
        tt = time.time() - self.total_train_time
        h, s = divmod(tt, 3600)
        m, s = divmod(s, 60)
        h, m = int(h), int(m)
        self.print('Total training time {h}:{m:>02}:{s:>05.2f}'.format(h=h, m=m, s=s))

    def get_metrics(self, trainer, val=False):
        metrics = trainer.callback_metrics
        if val:
            metrics = {k: v for k, v in metrics.items() if k.startswith('val/')}
        else:
            metrics = {k: v for k, v in metrics.items() if not k.startswith('val/')}
        for k in metrics:
            v = metrics[k]
            try:
                metrics[k] = round(v.item(), 4)
            except: pass
        return metrics
