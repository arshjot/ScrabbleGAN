
import torch
import torch.utils.data as data_utils
import torchvision.transforms as transforms
import pickle as pkl

from utils.data_utils import *

import pytorch_lightning as pl


class GANDataModule(pl.LightningDataModule):

    class CustomDataset(data_utils.Dataset):
        """
        Custom dataset
        Arguments:
            @config: 
        Returns:
        """

        def __init__(self, config, is_training=True):
            self.config = config
            self.is_training = is_training

            with open(config.data_file, 'rb') as f:
                data = pkl.load(f)

            self.word_data = data['word_data']
            self.idx_to_id = {i: w_id for i,
                              w_id in enumerate(self.word_data.keys())}
            self.char_map = data['char_map']

            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])

        def __len__(self):
            return len(self.word_data)

        def __getitem__(self, idx):
            item = {}
            w_id = self.idx_to_id[idx]

            # Get image and label
            lab, img = self.word_data[w_id]

            img = self.transforms(img / 255.)

            item['img'] = img.float()
            item['label'] = torch.tensor(lab)

            return item

    def __init__(self, config, is_training=True):
        super().__init__()
        self.config = config
        self.dataset = self.CustomDataset(config)

    def batch_collate(self, batch):
        items = {}
        max_w = max([item['img'].shape[2] for item in batch])

        # Remove channel dimension, swap height and width, pad widths and return to the original shape
        items['img'] = pad_sequence([item['img'].squeeze().permute(1, 0) for item in batch],
                                    batch_first=True,
                                    padding_value=1.)
        items['img'] = items['img'].permute(0, 2, 1).unsqueeze(1)

        items['label_len'] = torch.tensor(
            [len(item['label']) for item in batch])
        items['label'] = pad_sequence(
            [item['label'] for item in batch], batch_first=True, padding_value=0)

        return items

    def train_dataloader(self):
        return data_utils.DataLoader(
            self.dataset, batch_size=self.config.batch_size, shuffle=True,
            num_workers=2, pin_memory=True, collate_fn=self.batch_collate)
