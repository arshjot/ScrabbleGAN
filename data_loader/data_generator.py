import sys

sys.path.extend(['..'])

import torch
import torch.utils.data
import torch.utils.data as data_utils
import pickle as pkl

from utils.data_utils import *


# Dataset (Input Pipeline)
class CustomDataset(data_utils.Dataset):
    # TODO: Implement custom dataset functionalitites
    """
    Custom dataset

    Arguments:

    Returns:
    """

    def __init__(self, config=None, is_training=True):

        self.config = config
        self.is_training = is_training

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

class DataLoader:
    # TODO: Implement custom data loader functionalitites
    def __init__(self, config):
        self.config = config

        # # load data
        # with open(f'{self.config.data_file}', 'rb') as f:
        #     data_dict = pkl.load(f)

    def create_train_loader(self):

        dataset = CustomDataset(config=self.config)
        return torch.utils.data.DataLoader(dataset=dataset, batch_size=self.config.batch_size, shuffle=True,
                                           num_workers=3, pin_memory=True)

    def create_val_loader(self):

        dataset = CustomDataset(config=self.config, is_training=False)
        return torch.utils.data.DataLoader(dataset=dataset, batch_size=self.config.batch_size, num_workers=3,
                                           pin_memory=True)

    def create_test_loader(self):

        dataset = CustomDataset(config=self.config, is_training=False)
        return torch.utils.data.DataLoader(dataset=dataset, batch_size=self.config.batch_size, num_workers=3,
                                           pin_memory=True)
