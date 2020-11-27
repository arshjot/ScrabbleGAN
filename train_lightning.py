import os

from models.ScrabbleGAN import ScrabbleGAN
import torch
import numpy as np

from data_loader.lightning_data import GANDataModule
from data_loader.data_generator import DataLoader
from utils.data_utils import *
from config import Config

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import CometLogger

logger = TensorBoardLogger('tb_logs', name='ScrabbleGAN')
pl.seed_everything(0)

if __name__ == "__main__":
    config = Config
    dl = DataLoader(config).create_train_loader()
    model = ScrabbleGAN(config, dl.dataset.char_map)
    trainer = pl.Trainer(tpu_cores=8, automatic_optimization=False, logger=logger)
    trainer.fit(model, dl)
