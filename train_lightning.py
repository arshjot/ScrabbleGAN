import os

from models.ScrabbleGAN import ScrabbleGAN
import torch
import numpy as np

from data_loader.lightning_data import GANDataModule
from data_loader.data_generator import DataLoader
from utils.data_utils import *
from config import Config

import pytorch_lightning as pl
from pytorch_lightning.loggers import CometLogger
comet_logger = CometLogger(
    api_key="pMUMdGjynk01yd8KUmu7qAtw8",
    workspace="nicholas-robertson",    
    project_name='ScrabbleGAN',  # Optional
    rest_api_key=os.environ.get('COMET_REST_API_KEY'),  # Optional
    experiment_name='nan_grad_test'  # Optional
)

seed = 0
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
np.random.seed(seed)


if __name__ == "__main__":
    config = Config
    dl = DataLoader(config).create_train_loader()
    model = ScrabbleGAN(config, dl.dataset.char_map)
    trainer = pl.Trainer(fast_dev_run=True, track_grad_norm=2, logger=comet_logger)
    trainer.fit(model, dl)
