# TODO: Add required loss functions
import torch
import torch.nn as nn
import numpy as np


class MSELoss(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, yhat, y):
        loss = torch.mean(
            self.mse(yhat, y).mean(1))
        return loss


class RMSELoss(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, yhat, y):
        loss = torch.mean(
            torch.sqrt(self.mse(yhat, y).mean(1)))
        return loss
