"""
FILENAME: Loss.py
DESCRIPTION: Loss function definitions.
@author: Jian Zhong
"""

import torch
from torch import nn


# Kullback-Leibler divergence formula for diagnomal gauss distribution
def unitgauss_kullback_leibler_divergence(
        gauss_paras    
):
    mu, sigma = gauss_paras
    d_kl = 0.5 * torch.sum(sigma**2 + mu**2 - 1 - 2 * torch.log(sigma), dim = -1)
    return d_kl
    

# nn.Module for Kullback-Leibler divergence with unit gauss distribution
class UnitGaussKullbackLeiblerDivergenceLoss(nn.Module):
    def forward(self, gauss_params):
        d_kl = torch.mean(unitgauss_kullback_leibler_divergence(gauss_params))
        return d_kl
    

# nn.Module for similarity loss wih Gaussian assumptions
class GaussSimilarityLoss(nn.Module):
    def __init__(
            self,
            gauss_sigma = 1.0,
    ):
        assert gauss_sigma > 0
        super().__init__()
        self.register_buffer("gauss_sigma", torch.tensor(gauss_sigma))

    def forward(self, x, x_hat):
        x = torch.flatten(x, start_dim = 1, end_dim = -1)
        x_hat = torch.flatten(x_hat, start_dim = 1, end_dim = -1)
        batch_loss = 1/(2* (self.gauss_sigma**2) ) * torch.sum( (x - x_hat)**2, dim = -1)
        loss = torch.mean(batch_loss)
        return loss

