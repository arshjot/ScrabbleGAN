import torch
import torch.nn as nn
import numpy as np


class HingeLoss(torch.nn.Module):
    def __init__(self, d_or_g):
        super(HingeLoss, self).__init__()
        self.d_or_g = d_or_g  # 'D' / 'G'

    def forward(self, output, r_or_f='real'):
        if self.d_or_g == 'D':
            if r_or_f == 'real':
                hinge_loss = 1 - output
            else:
                hinge_loss = 1 + output
            hinge_loss[hinge_loss < 0] = 0
        else:
            hinge_loss = -output
        return hinge_loss.mean()


class CTCLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CTCLoss(blank=0, reduction='none', zero_infinity=True)

    def forward(self, input, target, input_lengths, target_lengths):
        loss = self.loss(input, target, input_lengths, target_lengths)
        return loss
