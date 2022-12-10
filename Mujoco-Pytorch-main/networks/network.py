import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.base import Network

import emlp.nn.pytorch as emlpnn
from emlp.reps import V
from emlp.groups import Z,S

from utils.utils import state2equistate
 
class Actor(Network):
    def __init__(self, layer_num, input_dim, output_dim, hidden_dim, activation_function = torch.tanh,last_activation = None, trainable_std = False):
        super(Actor, self).__init__(layer_num, input_dim, output_dim, hidden_dim, activation_function ,last_activation)
        G = Z(2)
        repin = 15 * V
        repout = 4 * V
        self.model = emlpnn.EMLP(repin, repout, G)
        self.trainable_std = trainable_std
       # self.parameters = self.model.parameters
        if self.trainable_std == True:
            self.logstd = self.model.parameters(torch.zeros(1, output_dim))

    def _forward(self,x):
        x = state2equistate(x)
        x = torch.tensor(x)
        x = self.model.forward(x)
        return x

    def forward(self, x):
        mu = self._forward(x)
        if self.trainable_std == True:
            std = torch.exp(self.logstd)
        else:
            logstd = torch.zeros_like(mu)
            std = torch.exp(logstd)
        return mu,std

class Critic(Network):
    def __init__(self, layer_num, input_dim, output_dim, hidden_dim, activation_function, last_activation = None):
        super(Critic, self).__init__(layer_num, input_dim, output_dim, hidden_dim, activation_function ,last_activation)
        G = Z(2)
        repin =  19 * V
        repout = 1 * V**0
        self.model = emlpnn.EMLP(repin, repout, G)
    def forward(self, *x):
        x = torch.cat(x,-1)
        return self._forward(x)
    