import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from importlib import import_module
import shutil
import glob
import os
import sys

from data_loader.data_generator import DataLoader
from utils.data_utils import *
from utils.training_utils import ModelCheckpoint, EarlyStopping
from losses_and_metrics import loss_functions, metrics
from config import Config

seed = 0
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
np.random.seed(seed)


class Trainer:
    def __init__(self, config):
        self.config = config
        self.terminal_width = shutil.get_terminal_size((80, 20)).columns

        # Model
        print(f' Model: {self.config.architecture} '.center(self.terminal_width, '*'))
        model_type = import_module('models.' + self.config.architecture)
        create_model = getattr(model_type, 'create_model')
        self.model = create_model(self.config)
        print(self.model, end='\n\n')

        # Loss, Optimizer and LRScheduler
        self.criterion = getattr(loss_functions, self.config.loss_fn)(self.config)
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.config.learning_rate, alpha=0.95)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5,
                                                                    patience=3, verbose=True)
        self.early_stopping = EarlyStopping(patience=10)
        
        # Metric
        self.metric = getattr(metrics, config.metric)()

        print(f' Loading Data '.center(self.terminal_width, '*'))
        data_loader = DataLoader(self.config)

        self.train_loader = data_loader.create_train_loader()
        self.val_loader = data_loader.create_val_loader()

        self.start_epoch, self.min_val_error = 1, None
        # Load checkpoint if training is to be resumed
        self.model_checkpoint = ModelCheckpoint(config=self.config)
        if config.resume_training:
            self.model, self.optimizer, self.scheduler, [self.start_epoch, self.min_val_error, num_bad_epochs] = \
                self.model_checkpoint.load(self.model, self.optimizer, self.scheduler)
            self.early_stopping.best = self.min_val_error
            self.early_stopping.num_bad_epochs = num_bad_epochs
            print(f'Resuming model training from epoch {self.start_epoch}')
        else:
            # remove previous logs, if any
            logs = glob.glob('./logs/.*') + glob.glob('./logs/*')
            for f in logs:
                try:
                    os.remove(f)
                except IsADirectoryError:
                    shutil.rmtree(f)

        # logging
        self.writer = SummaryWriter(f'logs')

    def _get_val_loss_and_err(self):
        # TODO: Complete validation loop to get validation loss and error
        self.model.eval()
        progbar = tqdm(self.val_loader)
        progbar.set_description("             ")
        losses, epoch_preds = [], []
        for i, [x, y] in enumerate(progbar):
            x = x.to(self.config.device)
            y = y.to(self.config.device)

            preds = self.model(x)
            epoch_preds.append(preds.data.cpu().numpy())
            loss = self.criterion(preds, y)
            losses.append(loss.data.cpu().numpy())

        # val_error = self.metric.get_error()
        val_error = np.random.random()
        val_loss = np.random.random()

        return val_loss, val_error

    def train(self):
        # TODO: Complete training loop
        print(f' Training '.center(self.terminal_width, '*'), end='\n\n')

        for epoch in range(self.start_epoch, self.config.num_epochs + 1):
            print(f' Epoch [{epoch}/{self.config.num_epochs}] '.center(self.terminal_width, 'x'))
            self.model.train()
            progbar = tqdm(self.train_loader)
            losses, epoch_preds = [], []

            for i, [x, y] in enumerate(progbar):
                x = x.to(self.config.device)
                y = y.to(self.config.device)

                # Forward + Backward + Optimize
                self.optimizer.zero_grad()
                preds = self.model(x)

                epoch_preds.append(preds.data.cpu().numpy())

                loss = self.criterion(preds, y)
                losses.append(loss.data.cpu().numpy())

                progbar.set_description("loss = %0.3f " % np.round(np.mean(losses), 3))

                loss.backward()
                self.optimizer.step()

            # Get training and validation loss and error
            # Sort to remove shuffle applied by the dataset loader
            # train_error = self.metric.get_error()
            train_error = np.random.random()
            train_loss = np.random.random()

            val_loss, val_error = self._get_val_loss_and_err()

            print(f'Training Loss: {train_loss:.4f}, Training Error: {train_error:.4f}, '
                  f'Training Secondary Error: {train_error_2:.4f}\n'
                  f'Validation Loss: {val_loss:.4f}, Validation Error: {val_error:.4f}, '
                  f'Validation Secondary Error: {val_error_2:.4f}')

            # Change learning rate according to scheduler
            self.scheduler.step(val_error)

            # save checkpoint and best model
            if self.min_val_error is None:
                self.min_val_error = val_error
                is_best = True
                print(f'Best model obtained at the end of epoch {epoch}')
            else:
                if val_error < self.min_val_error:
                    self.min_val_error = val_error
                    is_best = True
                    print(f'Best model obtained at the end of epoch {epoch}')
                else:
                    is_best = False
            self.model_checkpoint.save(is_best, self.min_val_error, self.early_stopping.num_bad_epochs,
                                       epoch, self.model, self.optimizer, self.scheduler)

            # write logs
            self.writer.add_scalar(f'{self.config.loss_fn}/train', train_loss, epoch * i)
            self.writer.add_scalar(f'{self.config.loss_fn}/val', val_loss, epoch * i)
            self.writer.add_scalar(f'{self.config.metric}/train', train_error, epoch * i)
            self.writer.add_scalar(f'{self.config.metric}/val', val_error, epoch * i)

            # Early Stopping
            if self.early_stopping.step(val_error):
                print(f' Training Stopped'.center(self.terminal_width, '*'))
                print(f'Early stopping triggered after epoch {epoch}')
                break

        self.writer.close()


if __name__ == "__main__":
    config = Config
    terminal_width = shutil.get_terminal_size((80, 20)).columns
    trainer = Trainer(config)
    trainer.train()
