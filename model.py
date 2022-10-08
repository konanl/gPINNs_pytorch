#!/usr/bin/env python
# encoding: utf-8
"""
    @author: LiangL. Yan
    @license: (C) Copyright forever
    @contact: yokingyan@gmail.com
    @ide: pycharm
    @file: model.py
    @time: 2022/9/26 23:40
    @desc:
    
"""
import torch
import torch.nn as nn
# from solver import *


class FCNet(nn.Module):
    """
    The basic full connected neural network.

    :parameter
        layers: The list of layers used in neural network
    """

    def __init__(self, layers, active=nn.Tanh()):
        """
        Initialization of FCNet
        :param
            layers: The list of layers used in neural network
            active: The activation function of neural network
        """
        super(FCNet, self).__init__()

        # Parameters
        self.depth = len(layers) - 1
        self.active = active

        # Layers list
        layer_list = list()
        for layer in range(self.depth - 1):
            layer_list.append(
                nn.Linear(layers[layer], layers[layer+1])
            )
            layer_list.append(active)
        layer_list.append(nn.Linear(layers[-2], layers[-1]))

        # Net
        self.main = nn.Sequential(*layer_list)

        # Initialize parameters
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize the param of network."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.main(x)


# util
def print_network(model, name):
    """Print out the information of network."""
    nums_params = 0
    for p in model.parameters():
        nums_params += p.numel()
    print(model)
    print(name)
    print("The number of parameters: {}".format(nums_params))


def print_net_params(model):
    """Print out the information of network params."""
    for parameter in model.parameters():
        print(parameter)
#


def gradients(x, y, order=1):
    """Computer the gradient : Dy/Dx."""
    if order == 1:
        return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),
                                   create_graph=True, retain_graph=True, only_inputs=True)[0]
    else:
        return gradients(gradients(x, y), x, order=order-1)


class PINNs(nn.Module):
    """
    The basic model of Physics Informed Neural Network.

    :parameter
        basic_net: The network structure of PINNs
             data: The dataset for training and testing PINNs
    """
    def __init__(self, basic_net, w=[1, 0], optimizer_="Adam"):
        """Init of PINNs."""
        super(PINNs, self).__init__()

        # NN
        self.net = basic_net

        # Optimizer
        self.optimizer = optimizer_

        # weight
        self.w_f, self.w_g = w[0], w[1]

    def output_transform(self, **kwargs):
        pass

    def forward(self, x):
        return self.net(x)

    def pde(self, **kwargs):
        """The basis PDE functions about every specific question."""
        return 0


if __name__ == '__main__':
    net = FCNet([1] + [20]*3 + [1])
    # print_network(net, 'FCNet')
    # print_net_params(net)
