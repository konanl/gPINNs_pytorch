#!/usr/bin/env python
# encoding: utf-8
"""
    @author: LiangL. Yan
    @license: (C) Copyright forever
    @contact: yokingyan@gmail.com
    @ide: pycharm
    @file: diffusion reaction.py
    @time: 2022/10/7 19:12
    @desc:
    
"""
import argparse
import os
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import visual_data
from visualization import *
from pde import *
from solver import *
from model import *
from process_data import *

pi = np.pi


def get_config():
    parser = argparse.ArgumentParser("PINNs/gPINNs for forward diffusion reaction model", add_help=False)

    # Training configurations.
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--net_type', default='pinn', type=str)
    parser.add_argument('--num_epochs', default=100000, type=int)
    parser.add_argument('--resume_epoch', type=int, default=0, help='resume training from this step')
    parser.add_argument('--num_epochs_decay', type=int, default=50000, help='decay lr from this step')
    parser.add_argument('--num_train_points', type=int, default=50, help='the number of training points')
    parser.add_argument('--g_weight', type=float, default=0.1)
    parser.add_argument('--output_transform', type=bool, default=True)
    parser.add_argument('--num_supervised_train_points', type=int, default=0,
                        help='the number of supervised training points')
    # Testing configurations.
    parser.add_argument('--test_epochs', type=int, default=80000, help='how long we should test model')
    parser.add_argument('--num_test_points', type=int, default=10000, help='the number of testing points')
    # Directories.
    parser.add_argument('--model_save_dir', type=str, default='./models/diffusion-reaction/model')
    parser.add_argument('--result_dir', type=str, default='./result')
    parser.add_argument('--log_dir', type=str, default='./models/diffusion-reaction/logs')
    # Step size.
    parser.add_argument('--log_step', type=int, default=5000)
    parser.add_argument('--model_save_step', type=int, default=20000)
    parser.add_argument('--lr_update_step', type=int, default=10000)

    return parser.parse_args()


class DiffusionReactionPINNs(PINNs):
    def __init__(self, net_, w=[1, 0]):
        super(DiffusionReactionPINNs, self).__init__(net_, w=w)

    def output_transform(self, x, y):
        x_in = x[:, 0:1]
        t_in = x[:, 1:2]

        return (x_in - pi) * (x_in + pi) * (1 - torch.exp(-t_in)) * y + icfunc(x_in)

    def pde(self, x):
        y = self.forward(x)
        y = self.output_transform(x, y)

        x_in = x[:, 0:1]
        t_in = x[:, 1:2]

        dy = gradients(x, y)
        d2y = gradients(x, dy)
        dydt = dy[:, 1:2]
        d2ydx2 = d2y[:, 0:1]

        r = torch.exp(-t_in) * (
                3 * torch.sin(2 * x_in) / 2
                + 8 * torch.sin(3 * x_in) / 3
                + 15 * torch.sin(4 * x_in) / 4
                + 63 * torch.sin(8 * x_in) / 8
        )
        eqs = dydt - d2ydx2 - r
        if self.w_g:
            d2ydtx = gradients(x, dydt)[:, 0:1]
            d3ydx3 = gradients(x, d2ydx2)[:, 0:1]
            drdx = torch.exp(-t_in) * (
                63 * torch.cos(8 * x_in)
                + 15 * torch.cos(4 * x_in)
                + 8 * torch.cos(3 * x_in)
                + 3 * torch.cos(2 * x_in)
            )

            d2yt2 = d2y[:, 1:2]
            d3ydx2t = gradients(x, d2y)[:, 1:2]
            drdt = -r
            g_eqs_x, g_eqs_t = d2ydtx - d3ydx3 - drdx, d2yt2 - d3ydx2t - drdt
        else:
            g_eqs_x = torch.zeros((1,), dtype=torch.float32)
            g_eqs_t = torch.zeros((1,), dtype=torch.float32)

        return [eqs, g_eqs_x, g_eqs_t]


def gen_all(num):
    """Generate the data."""
    xvals = np.linspace(-pi, pi, num, dtype=np.float32)[:, None]
    tvals = np.linspace(0, 1, num, dtype=np.float32)[:, None]
    a = np.hstack((xvals, tvals))
    yvals = diffusion_reaction_sol(a)
    y_0 = np.zeros(num, dtype=np.float32)[:, None]
    yvals = np.hstack((yvals, y_0))
    return np.reshape(a, (-1, 2)), np.reshape(yvals, (-1, 2))


def gen_test(num):
    x = np.linspace(-pi, pi, num, dtype=np.float32)
    t = np.linspace(0, 1, num, dtype=np.float32)
    X = []

    for i in range(len(t)):
        for j in range(len(x)):
            X.append([x[j], t[i]])

    return np.array(X)


if __name__ == '__main__':

    config = get_config()

    # Create directories if not exist.
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)

    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)

    # sys.stdout = visual_data.Logger(
    #     os.path.join(
    #         config.log_dir,
    #         'train-{}-{}.log'.format(
    #             "NN" if config.net_type == 'pinn' else "gNN, w={}".format(config.g_weight),
    #             config.num_train_points
    #         )
    #     ),
    #     sys.stdout
    # )

    print(config)

    # Train data.
    # train_x / valid_x -- [x, t]
    # train_x, train_u = gen_all(config.num_train_points)
    # # valid_x, valid_u = gen_all(config.num_test_points)
    # test_x = gen_test(100)
    # test_u = diffusion_reaction_sol(test_x)

    # train_data = np.stack((train_x, train_u))

    # Model
    # PINNs/gPINNs
    layers = [2] + [20] * 3 + [1]
    net_NN = FCNet(layers)
    model_NN = DiffusionReactionPINNs(net_NN)

    if config.net_type == 'gpinn':
        model_NN.w_g = config.g_weight

    # solver_NN = Solver(train_data, model_NN, config)
    # solver_NN.train()

    # Plot.
    # Exact
    # visual_data.plot_colormap(test_x, test_u,
    #                           't', 'x', 'Exact', './result/figure/diffusion-reaction/exact.png')

    # PINNs/gPINNs pred
    # u_pred = solver_NN.inference(test_x)[0]
    # visual_data.plot_colormap(
    #     test_x, u_pred.detach().numpy(),
    #     't', 'x', "{} Predict".format("PINNs" if config.net_type == "pinn" else "gPINNs"),
    #     './result/figure/diffusion-reaction/{}-{}.png'.format(
    #         "PINNs" if config.net_type == "pinn" else "gPINNs, w={}".format(config.g_weight), config.num_train_points
    #     )
    # )

    # absolute error
    # error = abs(test_u - u_pred.detach().numpy())
    # visual_data.plot_colormap(
    #     test_x, error,
    #     't', 'x', "Absolute Error",
    #     './result/figure/diffusion-reaction/absolute error-{}-{}.png'.format(
    #         "PINNs" if config.net_type == "pinn" else "gPINNs, w={}".format(config.g_weight), config.num_train_points
    #     )
    # )

    # Plot figure.3 ABCD
    L2_u = {}
    L2_dx = {}
    L2_dt = {}
    PDE_residual = {}

    training_points = [20, 30, 40, 50, 60, 70, 100, 120, 150]

    for training_point in training_points:

        training_point = int(training_point)

        config.num_train_points = training_point

        # Data
        train_x, train_u = gen_all(config.num_train_points)

        train_data = np.stack((train_x, train_u))

        test_x = gen_test(100)
        test_u = diffusion_reaction_sol(test_x)

        # dx/dt
        du_dx_true = dudx(test_x)
        du_dt_true = dudt(test_x)

        solver_NN = Solver(train_data, model_NN, config)

        # Train
        solver_NN.train()

        # Predict
        u_pred = solver_NN.inference(test_x)[0]
        dx_pred = solver_NN.inference(test_x)[1][:, 0:1]
        dt_pred = solver_NN.inference(test_x)[1][:, 1:2]

        # Error
        l2_u = ((test_u - u_pred.detach().numpy()) ** 2).mean()
        l2_dx = ((du_dx_true - dx_pred.detach().numpy())**2).mean()
        l2_dt = ((du_dt_true - dt_pred.detach().numpy())**2).mean()
        pde_res = ((model_NN.pde(torch.tensor(test_x, requires_grad=True))[0]) ** 2).mean()

        # Save loss dict
        L2_u['training point-{}'.format(config.num_train_points)] = l2_u
        L2_dx['training point-{}'.format(config.num_train_points)] = l2_dx
        L2_dt['training point-{}'.format(config.num_train_points)] = l2_dt
        PDE_residual['training point-{}'.format(config.num_train_points)] = pde_res

    # Save L2 relative error of u/u` and PDE residual
    save_dir = os.path.join(config.result_dir, 'loss/diffusion reaction')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    n = random.randint(1, 10)
    torch.save(L2_u, os.path.join(save_dir, 'l2_u-{}-{}'.format(
        'pinn' if config.net_type == "pinn" else "gpinn-w_{}".format(config.g_weight), n)))
    torch.save(L2_dx, os.path.join(save_dir, 'l2_dx-{}-{}'.format(
        'pinn'if config.net_type == "pinn" else "gpinn-w_{}".format(config.g_weight), n)))
    torch.save(L2_dt, os.path.join(save_dir, 'l2_dt-{}-{}'.format(
        'pinn'if config.net_type == "pinn" else "gpinn-w_{}".format(config.g_weight), n)))
    torch.save(PDE_residual, os.path.join(save_dir, 'pde_res-{}-{}'.format(
        'pinn' if config.net_type == "pinn" else "gpinn-w_{}".format(config.g_weight), n)))






