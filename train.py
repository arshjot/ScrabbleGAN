import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from importlib import import_module
import shutil
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import logging

import torch.nn.functional as F
from data_loader.data_generator import DataLoader
from utils.data_utils import *
from utils.training_utils import ModelCheckpoint
from losses_and_metrics import loss_functions, metrics
from config import Config

seed = 0
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
np.random.seed(seed)

level = logging.INFO
format_log = '%(message)s'
handlers = [logging.FileHandler('./output/output.log'), logging.StreamHandler()]
logging.basicConfig(level=level, format=format_log, handlers=handlers)


class Trainer:
    def __init__(self, config):
        self.config = config
        self.terminal_width = shutil.get_terminal_size((80, 20)).columns

        logging.info(f' Loading Data '.center(self.terminal_width, '*'))
        data_loader = DataLoader(self.config)

        self.train_loader = data_loader.create_train_loader()

        # Model
        logging.info(f' Model: {self.config.architecture} '.center(self.terminal_width, '*'))
        model_type = import_module('models.' + self.config.architecture)
        create_model = getattr(model_type, 'create_model')
        self.model = create_model(self.config, data_loader.dataset.char_map)
        logging.info(f'{self.model}\n')
        self.model.to(self.config.device)

        self.word_map = WordMap(data_loader.dataset.char_map)

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
            if epoch > (epoch - self.config.epochs_lr_decay - 1) else 1.
        self.schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, lr_decay_lambda) for opt in self.optimizers]

        # Metric
        # self.metric = getattr(metrics, config.metric)()

        self.start_epoch = 1
        # Load checkpoint if training is to be resumed
        self.model_checkpoint = ModelCheckpoint(config=self.config)
        if config.resume_training:
            self.model, self.optimizers, self.schedulers, self.start_epoch = \
                self.model_checkpoint.load(self.model, self.config.start_epoch, self.optimizers, self.schedulers)
            self.G_optimizer, self.D_optimizer, self.R_optimizer = self.optimizers
            logging.info(f'Resuming model training from epoch {self.start_epoch}')
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
        self.pred_R_real = self.model.R(self.real_img).permute(1, 0, 2)  # [w, b, num_chars]
        self.loss_R_real = self.R_criterion(self.pred_R_real, self.real_y,
                                            torch.ones(self.pred_R_real.size(1)).int() * self.pred_R_real.size(0),
                                            self.real_y_lens)
        self.loss_R_real = torch.mean(self.loss_R_real[~torch.isnan(self.loss_R_real)])

        self.loss_D_and_R = self.loss_D + self.loss_R_real

        self.loss_D_and_R.backward()

        self.D_optimizer.step()
        self.R_optimizer.step()
        self.D_optimizer.zero_grad()
        self.R_optimizer.zero_grad()

    def train(self):
        logging.info(f' Training '.center(self.terminal_width, '*'))

        for epoch in range(self.start_epoch, self.config.num_epochs + 1):
            logging.info(f' Epoch [{epoch}/{self.config.num_epochs}] '.center(self.terminal_width, 'x'))
            self.model.train()
            progbar = tqdm(self.train_loader)
            losses_G, losses_D, losses_D_real, losses_D_fake = [], [], [], []
            losses_R_real, losses_R_fake, grads_fake_R, grads_fake_adv = [], [], [], []

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

                # save losses
                losses_G.append(self.loss_G.cpu().data.numpy())
                losses_D.append(self.loss_D.cpu().data.numpy())
                losses_D_real.append(self.loss_D_real.cpu().data.numpy())
                losses_D_fake.append(self.loss_D_fake.cpu().data.numpy())
                losses_R_real.append(self.loss_R_real.cpu().data.numpy())
                losses_R_fake.append(self.loss_R_fake.cpu().data.numpy())
                grads_fake_R.append(self.loss_grad_fake_R.cpu().data.numpy())
                grads_fake_adv.append(self.loss_grad_fake_adv.cpu().data.numpy())

                progbar.set_description("G = %0.3f, D = %0.3f, R_real = %0.3f, R_fake = %0.3f,  " %
                                        (np.mean(losses_G), np.mean(losses_D),
                                         np.mean(losses_R_real), np.mean(losses_R_fake)))

            logging.info(f'G = {np.mean(losses_G):.3f}, D = {np.mean(losses_D):.3f}, '
                         f'R_real = {np.mean(losses_R_real):.3f}, R_fake = {np.mean(losses_R_fake):.3f}'
            )
            # Save one generated fake image from last batch
            plt.imshow(self.model.fake_img.cpu().data.numpy()[0][0], cmap='gray')
            plt.title(self.word_map.decode(self.model.fake_y.cpu().data.numpy()[0].reshape(1, -1)))
            plt.savefig(f'./output/epoch_{epoch}_fake_img.png')

            # Print Recognizer prediction for 4 (or batch size) real images from last batch
            num_imgs = 4 if self.config.batch_size >= 4 else self.config.batch_size
            labels = self.word_map.decode(self.real_y[:num_imgs].cpu().numpy())
            preds = self.word_map.recognizer_decode(self.pred_R_real.max(2)[1].permute(1, 0)[:num_imgs].cpu().numpy())
            logging.info('\nRecognizer predictions for real images:')
            max_len_label = max([len(i) for i in labels])
            for lab, pred in zip(labels, preds):
                logging.info(f'Actual: {lab:<{max_len_label+2}}|  Predicted: {pred}')

            # Print Recognizer prediction for 4 (or batch size) fake images from last batch
            logging.info('Recognizer predictions for fake images:')
            labels = self.word_map.decode(self.model.fake_y[:num_imgs].cpu().numpy())
            preds_R_fake = self.model.R(self.model.fake_img).permute(1, 0, 2).max(2)[1].permute(1, 0)
            preds = self.word_map.recognizer_decode(preds_R_fake[:num_imgs].cpu().numpy())
            max_len_label = max([len(i) for i in labels])
            for lab, pred in zip(labels, preds):
                logging.info(f'Actual: {lab:<{max_len_label+2}}|  Predicted: {pred}')

            # Change learning rate according to scheduler
            for sch in self.schedulers:
                sch.step()

            # save latest checkpoint and after every 5 epochs
            self.model_checkpoint.save(self.model, epoch, self.G_optimizer, self.D_optimizer, self.R_optimizer,
                                       *self.schedulers)
            if epoch > 1:
                if (epoch % 5) != 1:
                    os.remove(os.path.join(self.model_checkpoint.weight_dir, f'model_checkpoint_epoch_{epoch-1}.pth.tar'))

            # write logs
            self.writer.add_scalar(f'loss_G', np.mean(losses_G), epoch * i)
            self.writer.add_scalar(f'loss_D/fake', np.mean(losses_D_fake), epoch * i)
            self.writer.add_scalar(f'loss_D/real', np.mean(losses_D_real), epoch * i)
            self.writer.add_scalar(f'loss_R/fake', np.mean(losses_R_fake), epoch * i)
            self.writer.add_scalar(f'loss_R/real', np.mean(losses_R_real), epoch * i)
            self.writer.add_scalar(f'grads/fake_R', np.mean(grads_fake_R), epoch * i)
            self.writer.add_scalar(f'grads/fake_adv', np.mean(grads_fake_adv), epoch * i)

        self.writer.close()


if __name__ == "__main__":
    config = Config
    terminal_width = shutil.get_terminal_size((80, 20)).columns
    trainer = Trainer(config)
    trainer.train()
