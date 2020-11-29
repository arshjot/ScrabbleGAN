import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from models.model_utils import BigGAN as BGAN
from utils.data_utils import *
import pandas as pd


class Recognizer(nn.Module):
    def __init__(self, cfg):
        super(Recognizer, self).__init__()

        input_size = 1
        conv_is = [1] + cfg.r_fs

        self.convs = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(conv_is[0], cfg.r_fs[0], kernel_size=cfg.r_ks[0], padding=cfg.r_pads[0]),
                nn.ReLU(True),
                nn.MaxPool2d(2)
            ),
            nn.Sequential(
                nn.Conv2d(conv_is[1], cfg.r_fs[1], kernel_size=cfg.r_ks[1], padding=cfg.r_pads[1]),
                nn.ReLU(True),
                nn.MaxPool2d(2)
            ),
            nn.Sequential(
                nn.Conv2d(conv_is[2], cfg.r_fs[2], kernel_size=cfg.r_ks[2], padding=cfg.r_pads[2]),
                nn.BatchNorm2d(cfg.r_fs[2]),
                nn.ReLU(True)
            ),
            nn.Sequential(
                nn.Conv2d(conv_is[3], cfg.r_fs[3], kernel_size=cfg.r_ks[3], padding=cfg.r_pads[3]),
                nn.ReLU(True),
                nn.MaxPool2d((2, 2), (2, 1), (0, 1))
            ),
            nn.Sequential(
                nn.Conv2d(conv_is[4], cfg.r_fs[4], kernel_size=cfg.r_ks[4], padding=cfg.r_pads[4]),
                nn.BatchNorm2d(cfg.r_fs[4]),
                nn.ReLU(True)
            ),
            nn.Sequential(
                nn.Conv2d(conv_is[5], cfg.r_fs[5], kernel_size=cfg.r_ks[5], padding=cfg.r_pads[5]),
                nn.ReLU(True),
                nn.MaxPool2d((2, 2), (2, 1), (0, 1))
            ),
            nn.Sequential(
                nn.Conv2d(conv_is[6], cfg.r_fs[6], kernel_size=cfg.r_ks[6], padding=cfg.r_pads[6]),
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


class ScrabbleGAN(nn.Module):
    def __init__(self, cfg, char_map):
        super().__init__()

        self.z_dist = torch.distributions.Normal(loc=0, scale=1.)
        self.z_dim = cfg.z_dim

        # Get word list from lexicon to be used to generate fake images
        if cfg.dataset == 'IAM':
            self.fake_words = pd.read_csv(cfg.lexicon_file, sep='\t', names=['words'])
            # filter words with len >= 20
            self.fake_words = self.fake_words.loc[self.fake_words.words.str.len() < 20]
            self.fake_words = self.fake_words.words.to_list()
        else:
            exception_chars = ['ï', 'ü', '.', '_', 'ö', ',', 'ã', 'ñ']
            self.fake_words = pd.read_csv(cfg.lexicon_file, '\t')['lemme']
            self.fake_words = [word.split()[-1] for word in self.fake_words
                               if (pd.notnull(word) and all(char not in word for char in exception_chars))]

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
        self.D = BGAN.Discriminator(resolution=cfg.resolution, bn_linear=cfg.bn_linear, n_classes=cfg.num_chars)

    def forward_fake(self, z=None, fake_y=None, b_size=None):
        b_size = self.batch_size if b_size is None else b_size

        # If z is not provided, sample it
        if z is None:
            self.z = self.z_dist.sample([b_size, self.z_dim]).to(self.config.device)
        else:
            self.z = z.repeat(b_size, 1).to(self.config.device)

        # If fake words are not provided, sample it
        if fake_y is None:
            # Sample lexicon indices, get words, and encode them using char_map
            sample_lex_idx = self.fake_y_dist.sample([b_size])
            fake_y = [self.fake_words[i] for i in sample_lex_idx]
            fake_y, fake_y_lens = self.word_map.encode(fake_y)
            self.fake_y_lens = fake_y_lens.to(self.config.device)

        # Convert y into one-hot
        self.fake_y = fake_y.to(self.config.device)
        self.fake_y_one_hot = F.one_hot(fake_y, self.num_chars).to(self.config.device)

        self.fake_img = self.G(self.z, self.fake_y_one_hot)


def create_model(config, char_map):
    model = ScrabbleGAN(config, char_map)
    model.to(config.device)

    return model
