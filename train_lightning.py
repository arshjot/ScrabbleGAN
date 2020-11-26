from models.ScrabbleGAN import ScrabbleGAN
import torch
import numpy as np

from data_loader.lightning_data import GANDataModule
from data_loader.data_generator import DataLoader
from utils.data_utils import *
from config import Config

import pytorch_lightning as pl

seed = 0
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
np.random.seed(seed)


if __name__ == "__main__":
    config = Config
    dl = DataLoader(config).create_train_loader()
    model = ScrabbleGAN(config, dl.dataset.char_map)
    trainer = pl.Trainer(max_epochs=5, progress_bar_refresh_rate=20)
    trainer.fit(model, dl)
