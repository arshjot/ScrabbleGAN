from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision

from models.model_utils import BigGAN as BGAN
from losses_and_metrics import loss_functions, metrics
from config import Config

from utils.data_utils import *
import pandas as pd
import numpy as np

import pytorch_lightning as pl


class Recognizer(nn.Module):
    def __init__(self, cfg):
        super(Recognizer, self).__init__()

        input_size = 1
        conv_is = [1] + cfg.r_fs

        self.convs = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(conv_is[0], cfg.r_fs[0],
                          kernel_size=cfg.r_ks[0], padding=cfg.r_pads[0]),
                nn.ReLU(True),
                nn.MaxPool2d(2)
            ),
            nn.Sequential(
                nn.Conv2d(conv_is[1], cfg.r_fs[1],
                          kernel_size=cfg.r_ks[1], padding=cfg.r_pads[1]),
                nn.ReLU(True),
                nn.MaxPool2d(2)
            ),
            nn.Sequential(
                nn.Conv2d(conv_is[2], cfg.r_fs[2],
                          kernel_size=cfg.r_ks[2], padding=cfg.r_pads[2]),
                nn.BatchNorm2d(cfg.r_fs[2]),
                nn.ReLU(True)
            ),
            nn.Sequential(
                nn.Conv2d(conv_is[3], cfg.r_fs[3],
                          kernel_size=cfg.r_ks[3], padding=cfg.r_pads[3]),
                nn.ReLU(True),
                nn.MaxPool2d((2, 2), (2, 1), (0, 1))
            ),
            nn.Sequential(
                nn.Conv2d(conv_is[4], cfg.r_fs[4],
                          kernel_size=cfg.r_ks[4], padding=cfg.r_pads[4]),
                nn.BatchNorm2d(cfg.r_fs[4]),
                nn.ReLU(True)
            ),
            nn.Sequential(
                nn.Conv2d(conv_is[5], cfg.r_fs[5],
                          kernel_size=cfg.r_ks[5], padding=cfg.r_pads[5]),
                nn.ReLU(True),
                nn.MaxPool2d((2, 2), (2, 1), (0, 1))
            ),
            nn.Sequential(
                nn.Conv2d(conv_is[6], cfg.r_fs[6],
                          kernel_size=cfg.r_ks[6], padding=cfg.r_pads[6]),
                nn.BatchNorm2d(cfg.r_fs[6]),
                nn.ReLU(True)
            )
        )

        self.output = nn.Linear(512, cfg.num_chars)
        self.prob = nn.LogSoftmax(dim=2)

    def forward(self, x):
        out = self.convs(x)

        out = out.squeeze(2)  # [b, c, w]
        out = out.permute(0, 2, 1)  # [b, w, c]

        # Predict for len(num_chars) classes at each timestep
        out = self.output(out)
        out = self.prob(out)
        return out


class ScrabbleGAN(pl.LightningModule):
    def __init__(self, cfg: Config, char_map):
        super().__init__()

        self.G_criterion = None
        self.D_criterion = None
        self.R_criterion = None

        self.z_dist = torch.distributions.Normal(loc=0, scale=1.)
        self.z_dim = cfg.z_dim

        # Get word list from lexicon to be used to generate fake images, filter words with len >= 20
        self.fake_words = pd.read_csv(
            cfg.lexicon_file, sep='\t', names=['words'])
        self.fake_words = self.fake_words.loc[self.fake_words.words.str.len(
        ) < 20]
        self.fake_words = self.fake_words.words.to_list()

        fake_words_clean = []
        for word in self.fake_words:
            word_set = set(word)
            if len(word_set.intersection(char_map.keys())) == len(word_set):
                fake_words_clean.append(word)
        self.fake_words = fake_words_clean

        self.fake_y_dist = torch.distributions.Categorical(
            torch.tensor([1. / len(self.fake_words)] * len(self.fake_words)))

        self.batch_size = cfg.batch_size
        self.num_chars = cfg.num_chars
        self.word_map = WordMap(char_map)

        self.batch_size = cfg.batch_size
        self.num_chars = cfg.num_chars
        self.config = cfg

        self.R = Recognizer(cfg)
        self.G = BGAN.Generator(resolution=cfg.resolution, G_shared=cfg.g_shared,
                                bn_linear=cfg.bn_linear, n_classes=cfg.num_chars, hier=True)
        self.D = BGAN.Discriminator(
            resolution=cfg.resolution, bn_linear=cfg.bn_linear, n_classes=cfg.num_chars)

    def forward(self, z, fake_y):
        return self.G(z, fake_y)

    def training_step(self, batch, batch_idx, optimizer_idx):

        imgs = batch['img']
        real_y = batch['label']
        real_y_one_hot = F.one_hot(real_y, self.config.num_chars)
        real_y_lens = batch['label_len']

        # sample noise
        z = self.z_dist.sample([self.config.batch_size, self.z_dim])
        z = z.type_as(imgs)
        sample_lex_idx = self.fake_y_dist.sample([self.batch_size])
        fake_y = [self.fake_words[i] for i in sample_lex_idx]
        fake_y, fake_y_lens = self.word_map.encode(fake_y)

        # train generator
        if optimizer_idx == 0:
            # generate images
            self.generated_imgs = self(z, fake_y)

            pred_D_fake = self.D(self.generated_imgs)
            pred_R_fake = self.R(self.generated_imgs).permute(
                1, 0, 2)  # [w, b, num_chars]

            valid = torch.ones(pred_R_fake.size(1)).int()
            valid = valid.type_as(imgs)

            loss_G = self.G_criterion(pred_D_fake)
            loss_R_fake = self.R_criterion(pred_R_fake, fake_y,
                                           valid * pred_R_fake.size(0),
                                           fake_y_lens)
            loss_R_fake = torch.mean(loss_R_fake[~torch.isnan(loss_R_fake)])

            loss_G_total = loss_G + self.config.grad_alpha * loss_R_fake
            grad_fake_R = torch.autograd.grad(
                loss_R_fake, self.generated_imgs, retain_graph=True)[0]
            grad_fake_adv = torch.autograd.grad(
                loss_G, self.generated_imgs, retain_graph=True)[0]
            epsilon = 10e-50
            a = self.config.grad_alpha * \
                torch.div(torch.std(grad_fake_adv),
                          epsilon + torch.std(grad_fake_R))
            loss_R_fake = a.detach() * loss_R_fake
            loss_G_total = loss_G + loss_R_fake

            # adversarial loss is binary cross-entropy
            g_loss = loss_G_total
            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        # train discriminator
        if optimizer_idx == 1:

            self.generated_imgs = self(z, fake_y)

            pred_D_fake = self.D(self.generated_imgs.detach())
            pred_D_real = self.D(imgs.detach())

            # we will now calculate discriminator loss for both real and fake images
            d_loss_fake = self.D_criterion(pred_D_fake, 'fake')
            d_loss_real = self.D_criterion(pred_D_real, 'real')
            d_loss = d_loss_fake + d_loss_real

            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        # train recogniser
        if optimizer_idx == 2:
            # recognizer
            pred_R_real = self.R(imgs).permute(1, 0, 2)  # [w, b, num_chars]

            valid = torch.ones(pred_R_real.size(1)).int()
            valid = valid.type_as(imgs)

            r_loss = self.R_criterion(pred_R_real, real_y,
                                      valid * pred_R_real.size(0),
                                      real_y_lens)
            r_loss = torch.mean(r_loss[~torch.isnan(r_loss)])

            tqdm_dict = {'r_loss': r_loss}
            output = OrderedDict({
                'loss': r_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

    def configure_optimizers(self):
        self.G_criterion = getattr(loss_functions, self.config.g_loss_fn)('G')
        self.D_criterion = getattr(loss_functions, self.config.d_loss_fn)('D')
        self.R_criterion = getattr(loss_functions, self.config.r_loss_fn)()
        G_optimizer = torch.optim.Adam(
            self.G.parameters(), lr=self.config.g_lr, betas=self.config.g_betas)
        D_optimizer = torch.optim.Adam(
            self.D.parameters(), lr=self.config.d_lr, betas=self.config.d_betas)
        R_optimizer = torch.optim.Adam(
            self.R.parameters(), lr=self.config.r_lr, betas=self.config.r_betas)

        self.optimizers = [G_optimizer, D_optimizer, R_optimizer]

        def lr_decay_lambda(epoch): return (1. - (1. / self.config.epochs_lr_decay)) \
            if epoch > (epoch - self.config.epochs_lr_decay) else 1.
        self.schedulers = [torch.optim.lr_scheduler.LambdaLR(
            opt, lr_decay_lambda) for opt in self.optimizers]

        return self.optimizers, self.schedulers

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_idx,
                       second_order_closure=None, on_tpu=False, using_native_amp=False, using_lbfgs=False):
        if optimizer_idx == 0:
            if batch_nb % self.config.train_gen_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        # update discriminator opt every 4 steps
        if optimizer_idx == 1:
            optimizer.step()
            optimizer.zero_grad()
        if optimizer_idx == 1:
            optimizer.step()
            optimizer.zero_grad()
