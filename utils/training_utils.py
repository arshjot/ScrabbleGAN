import torch
import shutil
import os
import numpy as np
from config import Config


class ModelCheckpoint:
    def __init__(self, weight_dir='./weights', config=Config):
        self.weight_dir = weight_dir
        self.config = config

    def save(self, model, epoch, G_opt, D_opt, R_opt, G_sch=None, D_sch=None, R_sch=None):
        filename = os.path.join(self.weight_dir, f'model_checkpoint_epoch_{epoch}.pth.tar')
        save_dict = {
            'model': model.state_dict(),
            'G_opt': G_opt.state_dict(),
            'D_opt': D_opt.state_dict(),
            'R_opt': R_opt.state_dict(),
            'epoch': epoch,
            'G_sch': G_sch.state_dict(),
            'D_sch': D_sch.state_dict(),
            'R_sch': R_sch.state_dict()
        }
        torch.save(save_dict, filename)

    def load(self, model, epoch, optimizers=None, schedulers=None, checkpoint_path=None):
        [G_opt, D_opt, R_opt] = optimizers if optimizers is not None else [None]*3
        [G_sch, D_sch, R_sch] = schedulers if schedulers is not None else [None]*3

        if checkpoint_path is None:
            load_filename = os.path.join(self.weight_dir, f'model_checkpoint_epoch_{epoch}.pth.tar')
        else:
            load_filename = checkpoint_path
        if os.path.isfile(load_filename):
            checkpoint = torch.load(load_filename, map_location=self.config.device)
            model.load_state_dict(checkpoint['model'])

            if optimizers is not None:
                G_opt.load_state_dict(checkpoint['G_opt'])
                D_opt.load_state_dict(checkpoint['D_opt'])
                R_opt.load_state_dict(checkpoint['R_opt'])
            if schedulers is not None:
                G_sch.load_state_dict(checkpoint['G_sch'])
                D_sch.load_state_dict(checkpoint['D_sch'])
                R_sch.load_state_dict(checkpoint['R_sch'])

            start_epoch = checkpoint['epoch'] + 1
        else:
            raise FileNotFoundError(f'No checkpoint found at {load_filename}')

        return model, [G_opt, D_opt, R_opt], [G_sch, D_sch, R_sch], start_epoch


class EarlyStopping(object):
    """
    author:https://github.com/stefanonardo
    source: https://gist.github.com/stefanonardo/693d96ceb2f531fa05db530f3e21517d
    """
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)
