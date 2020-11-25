import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from importlib import import_module
import shutil
import glob
import os
import numpy as np
import sys

import torch.nn.functional as F
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

        print(f' Loading Data '.center(self.terminal_width, '*'))
        data_loader = DataLoader(self.config)

        self.train_loader = data_loader.create_train_loader()

        # Model
        print(f' Model: {self.config.architecture} '.center(self.terminal_width, '*'))
        model_type = import_module('models.' + self.config.architecture)
        create_model = getattr(model_type, 'create_model')
        self.model = create_model(self.config, data_loader.dataset.char_map)
        print(self.model, end='\n\n')
        self.model.to(self.config.device)

        # Loss, Optimizer and LRScheduler
        self.G_criterion = getattr(loss_functions, self.config.g_loss_fn)('G')
        self.D_criterion = getattr(loss_functions, self.config.d_loss_fn)('D')
        self.R_criterion = getattr(loss_functions, self.config.r_loss_fn)()
        self.G_optimizer = torch.optim.Adam(self.model.G.parameters(), lr=self.config.g_lr, betas=self.config.g_betas)
        self.D_optimizer = torch.optim.Adam(self.model.D.parameters(), lr=self.config.d_lr, betas=self.config.d_betas)
        self.R_optimizer = torch.optim.Adam(self.model.R.parameters(), lr=self.config.r_lr, betas=self.config.r_betas)
        self.optimizers = [self.G_optimizer, self.D_optimizer, self.R_optimizer]

        # Use a linear learning rate decay but start the decay only after specified number of epochs
        lr_decay_lambda = lambda epoch: (1. - (1. / self.config.epochs_lr_decay)) \
            if epoch > (epoch - self.config.epochs_lr_decay) else 1.
        self.schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, lr_decay_lambda) for opt in self.optimizers]
        # self.early_stopping = EarlyStopping(patience=10)
        
        # Metric
        # self.metric = getattr(metrics, config.metric)()

        self.start_epoch, self.min_val_error = 1, None
        # Load checkpoint if training is to be resumed
        self.model_checkpoint = ModelCheckpoint(config=self.config)
        if config.resume_training:
            self.model, self.optimizers, self.schedulers, [self.start_epoch, self.min_val_error, num_bad_epochs] = \
                self.model_checkpoint.load(self.model, self.optimizers, self.schedulers)
            [self.G_optimizer, self.D_optimizer, self.R_optimizer] = self.optimizers
            # self.early_stopping.best = self.min_val_error
            # self.early_stopping.num_bad_epochs = num_bad_epochs
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

    def set_requires_grad(self, nets, requires_grad=False):
        """
        Source - https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/fd29199c33bd95704690aaa16f238a4f8e74762c/models/base_model.py
        Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def optimize_G(self):
        """Completes forward, backward, and optimize for G"""
        # generate fake image using generator
        self.model.forward_fake()
        # Switch off backpropagation for R and D
        self.set_requires_grad([self.model.D, self.model.R], False)

        # Generator loss will be determined by the evaluation of generated image by discriminator and recognizer
        pred_D_fake = self.model.D(self.model.fake_img)
        pred_R_fake = self.model.R(self.model.fake_img).permute(1, 0, 2)  # [w, b, num_chars]

        self.loss_G = self.G_criterion(pred_D_fake)
        self.loss_R_fake = self.R_criterion(pred_R_fake, self.model.fake_y,
                                            torch.ones(pred_R_fake.size(1)).int() * pred_R_fake.size(0),
                                            self.model.fake_y_lens)
        self.loss_R_fake = torch.mean(self.loss_R_fake[~torch.isnan(self.loss_R_fake)])

        # the below part has been mostly copied from - https://github.com/amzn/convolutional-handwriting-gan/blob/2cfbc794cca299445e5ba070c8634b6cd1a84261/models/ScrabbleGAN_baseModel.py#L345
        self.loss_G_total = self.loss_G + self.config.grad_alpha * self.loss_R_fake
        grad_fake_R = torch.autograd.grad(self.loss_R_fake, self.model.fake_img, retain_graph=True)[0]
        self.loss_grad_fake_R = 10 ** 6 * torch.mean(grad_fake_R ** 2)
        grad_fake_adv = torch.autograd.grad(self.loss_G, self.model.fake_img, retain_graph=True)[0]
        self.loss_grad_fake_adv = 10 ** 6 * torch.mean(grad_fake_adv ** 2)
        if self.config.grad_balance:
            epsilon = 10e-50
            self.loss_G_total.backward(retain_graph=True)
            grad_fake_R = torch.autograd.grad(self.loss_R_fake, self.model.fake_img,
                                              create_graph=True, retain_graph=True)[0]
            grad_fake_adv = torch.autograd.grad(self.loss_G, self.model.fake_img,
                                                create_graph=True, retain_graph=True)[0]
            a = self.config.grad_alpha * torch.div(torch.std(grad_fake_adv), epsilon + torch.std(grad_fake_R))
            self.loss_R_fake = a.detach() * self.loss_R_fake
            self.loss_G_total = self.loss_G + self.loss_R_fake
            self.loss_G_total.backward(retain_graph=True)
            grad_fake_R = torch.autograd.grad(self.loss_R_fake, self.model.fake_img,
                                              create_graph=False, retain_graph=True)[0]
            grad_fake_adv = torch.autograd.grad(self.loss_G, self.model.fake_img,
                                                create_graph=False, retain_graph=True)[0]
            self.loss_grad_fake_R = 10 ** 6 * torch.mean(grad_fake_R ** 2)
            self.loss_grad_fake_adv = 10 ** 6 * torch.mean(grad_fake_adv ** 2)
            with torch.no_grad():
                self.loss_G_total.backward()
        else:
            self.loss_G_total.backward()

        self.G_optimizer.step()
        self.G_optimizer.zero_grad()

    def optimize_D_R(self):
        """Completes forward, backward, and optimize for D and R"""
        # generate fake image using generator
        self.model.forward_fake()
        # Switch on backpropagation for R and D
        self.set_requires_grad([self.model.D, self.model.R], True)

        pred_D_fake = self.model.D(self.model.fake_img.detach())
        pred_D_real = self.model.D(self.real_img.detach())

        # we will now calculate discriminator loss for both real and fake images
        self.loss_D_fake = self.D_criterion(pred_D_fake, 'fake')
        self.loss_D_real = self.D_criterion(pred_D_real, 'real')
        self.loss_D = self.loss_D_fake + self.loss_D_real

        # recognizer
        pred_R_real = self.model.R(self.real_img).permute(1, 0, 2)  # [w, b, num_chars]
        self.loss_R_real = self.R_criterion(pred_R_real, self.real_y,
                                            torch.ones(pred_R_real.size(1)).int() * pred_R_real.size(0),
                                            self.real_y_lens)
        self.loss_R_real = torch.mean(self.loss_R_real[~torch.isnan(self.loss_R_real)])

        self.loss_D_and_R = self.loss_D + self.loss_R_real

        self.loss_D_and_R.backward()

        self.D_optimizer.step()
        self.R_optimizer.step()
        self.D_optimizer.zero_grad()
        self.R_optimizer.zero_grad()


    def train(self):
        print(f' Training '.center(self.terminal_width, '*'), end='\n\n')

        for epoch in range(self.start_epoch, self.config.num_epochs + 1):
            print(f' Epoch [{epoch}/{self.config.num_epochs}] '.center(self.terminal_width, 'x'))
            self.model.train()
            progbar = tqdm(self.train_loader)
            losses, epoch_preds = [], []

            for i, batch_items in enumerate(progbar):
                self.real_img = batch_items['img'].to(self.config.device)
                self.real_y = batch_items['label'].to(self.config.device)
                self.real_y_one_hot = F.one_hot(batch_items['label'], self.config.num_chars).to(self.config.device)
                self.real_y_lens = batch_items['label_len'].to(self.config.device)

                # Forward + Backward + Optimize G
                if (i % self.config.train_gen_steps) == 0:
                    # optimize generator
                    self.optimize_G()

                # Forward + Backward + Optimize D and R
                self.optimize_D_R()

                # epoch_preds.append(preds.data.cpu().numpy())
                #
                # loss = self.criterion(preds, y)
                # losses.append(loss.data.cpu().numpy())
                #
                progbar.set_description("G = %0.3f, D = %0.3f, R_real = %0.3f, R_fake = %0.3f,  " % (self.loss_G, self.loss_D, self.loss_R_real, self.loss_R_fake))

            # Change learning rate according to scheduler
            for sch in self.schedulers:
                sch.step(epoch)

            # # save checkpoint
            # if self.min_val_error is None:
            #     self.min_val_error = val_error
            #     is_best = True
            #     print(f'Best model obtained at the end of epoch {epoch}')
            # else:
            #     if val_error < self.min_val_error:
            #         self.min_val_error = val_error
            #         is_best = True
            #         print(f'Best model obtained at the end of epoch {epoch}')
            #     else:
            #         is_best = False
            # self.model_checkpoint.save(is_best, self.min_val_error, self.early_stopping.num_bad_epochs,
            #                            epoch, self.model, [self.G_optimizer, self.D_optimizer, self.R_optimizer],
            #                            self.scheduler)
            #
            # # write logs
            # self.writer.add_scalar(f'{self.config.loss_fn}/train', train_loss, epoch * i)
            # self.writer.add_scalar(f'{self.config.loss_fn}/val', val_loss, epoch * i)
            # self.writer.add_scalar(f'{self.config.metric}/train', train_error, epoch * i)
            # self.writer.add_scalar(f'{self.config.metric}/val', val_error, epoch * i)
            #
            # # Early Stopping
            # if self.early_stopping.step(val_error):
            #     print(f' Training Stopped'.center(self.terminal_width, '*'))
            #     print(f'Early stopping triggered after epoch {epoch}')
            #     break

        self.writer.close()


if __name__ == "__main__":
    config = Config
    terminal_width = shutil.get_terminal_size((80, 20)).columns
    trainer = Trainer(config)
    trainer.train()
