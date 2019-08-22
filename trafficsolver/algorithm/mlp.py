import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def init_weights(m, gain):
    if (type(m) == nn.Linear) | (type(m) == nn.Conv2d):
        nn.init.orthogonal_(m.weight, gain)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class MLP(torch.nn.Module):
    """
    MLP with ReLU activations after each hidden layer, but not on the output layer
    
    n_output can be None, in that case there is no output layer and so the last layer
    is the last hidden layer, with a ReLU
    """

    def __init__(self, observation_space, n_outputs, hiddens=[100, 100],**kwargs):
        super().__init__()

        if len(observation_space.shape) != 1:
            raise NotImplementedError
        else:
            n_inputs = observation_space.shape[0]

        layers = []
        for hidden in hiddens:
            layers.append(nn.Linear(n_inputs, hidden))
            layers.append(nn.ReLU())
            n_inputs = hidden

        if n_outputs is not None:
            layers.append(nn.Linear(hidden, n_outputs))

        self.layers = nn.Sequential(*layers)
        self.layers.apply(lambda x: init_weights(x, 3))

    def forward(self, obs):
        return self.layers(obs)
