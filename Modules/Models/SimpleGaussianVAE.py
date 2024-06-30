"""
FILENAME: SimpleGaussianVAE.py
DESCRIPTION: Definition for simple (diagonal) gaussian variant autoencoder network.
@author: Jian Zhong
"""

import torch
from torch import nn

from .Layers import (DebugLayers, StackedLayers)


# Encoder learns mappling from input data x to latent z gaussian distribution parameters (i.e. mu and sigma)
class SimpleGaussVAEFCEncoder(nn.Module):
    def __init__(
        self,
        prev_layer_descriptors = [],
        gaussparam_layer_descriptors = [],
    ):
        assert isinstance(prev_layer_descriptors, list)
        assert isinstance(gaussparam_layer_descriptors, list)
        super().__init__()

        self.prev_network = StackedLayers.VGGStackedLinear(prev_layer_descriptors)

        in_features_key = r"in_features"
        if in_features_key not in gaussparam_layer_descriptors[0]:
            last_fc_idx = len(self.prev_network.network) - 1
            while(last_fc_idx > 0):
                if(isinstance(self.prev_network.network[last_fc_idx], nn.Linear)):
                    break
                last_fc_idx -= 1
            gaussparam_layer_descriptors[0][in_features_key] = self.prev_network.network[last_fc_idx].weight.size(0)

        self.gauss_mu_network = StackedLayers.VGGStackedLinear(gaussparam_layer_descriptors)
        self.gauss_logsigma_network = StackedLayers.VGGStackedLinear(gaussparam_layer_descriptors)

    def forward(self, x):
        prev_y = self.prev_network(x)
        gauss_mu = self.gauss_mu_network(prev_y)
        gauss_sigma = torch.exp(self.gauss_logsigma_network(prev_y))
        return (gauss_mu, gauss_sigma)


# generate diagonal gaussian random variables
class DiagGaussSample(nn.Module):
    def __init__(self):
        super().__init__()
        self.unit_normal = torch.distributions.normal.Normal(0, 1)

    def forward(self, gauss_paras):
        mu, sigma = gauss_paras
        normal_samples = self.unit_normal.sample(sigma.size()).to(sigma.device)
        z = mu + sigma * normal_samples
        return z


# # generate diagonal gaussian random variables
# class DiagGaussSample(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.register_buffer("gauss_mu", torch.tensor(0.0)) 
#         self.register_buffer("gauss_sigma", torch.tensor(1.0))

#     def forward(self, gauss_params):
#         mu, sigma = gauss_params
#         z = mu + sigma * torch.normal(self.gauss_mu, self.gauss_sigma, size = sigma.size())
#         return z
        
        

# decoder learns mapping from latent variable z to reconstructed data x_hat 
class SimpleGaussVAEFCDecoder(nn.Module):
    def __init__(
        self,
        layer_descriptors = [],
    ):
        assert isinstance(layer_descriptors, list)
        super().__init__()
        self.network = StackedLayers.VGGStackedLinear(layer_descriptors)
    
    def forward(self, x):
        y = self.network(x)
        return y
