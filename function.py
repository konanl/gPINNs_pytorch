#!/usr/bin/env python
# encoding: utf-8
"""
    @author: LiangL. Yan
    @license: (C) Copyright forever
    @contact: yokingyan@gmail.com
    @ide: pycharm
    @file: function.py
    @time: 2022/10/7 11:11
    @desc:
    
"""
import argparse
import os.path
import random

import torch

import visual_data
from visualization import *
from pde import *
from solver import *
from model import *
from process_data import *


def get_config():
    parser = argparse.ArgumentParser("PINNs for function", add_help=False)

    # Training configurations.
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--net_type', default='pinn', type=str)
    parser.add_argument('--num_epochs', default=10000, type=int)
    parser.add_argument('--resume_epoch', type=int, default=0, help='resume training from this step')
    parser.add_argument('--num_epochs_decay', type=int, default=5000, help='decay lr from this step')
    parser.add_argument('--num_train_points', type=int, default=15, help='the number of training points')
    parser.add_argument('--g_weight', type=float, default=1)
    parser.add_argument('--output_transform', type=bool, default=False)
    parser.add_argument('--num_supervised_train_points', type=int, default=0,
                        help='the number of supervised training points')
    # Testing configurations.
    parser.add_argument('--test_epochs', type=int, default=2000, help='how long we should test model')
    parser.add_argument('--num_test_points', type=int, default=1000, help='the number of testing points')
    # Directories.
    parser.add_argument('--model_save_dir', type=str, default='./models/function/model')
    parser.add_argument('--result_dir', type=str, default='./result')
    parser.add_argument('--log_dir', type=str, default='./models/function/logs')
    # Step size.
    parser.add_argument('--log_step', type=int, default=2000)
    parser.add_argument('--model_save_step', type=int, default=2000)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    return parser.parse_args()


class FunctionPINNs(PINNs):
    def __init__(self, net_, w=[1, 0]):
        super(FunctionPINNs, self).__init__(net_, w=w)

    def output_transform(self, x, y):
        return y

    def pde(self, x):
        y = self.forward(x)
        eqs = y + (1.4 - 3 * x) * torch.sin(18 * x)
        if self.w_g:
            g_eqs = gradients(x, eqs)
        else:
            g_eqs = torch.zeros((1,), dtype=torch.float32)
        return [eqs, g_eqs]


def gen_all(num):
    xvals = np.linspace(0, 1, num+2, dtype=np.float32)[1:-1, None]
    yvals = function_sol(xvals)
    ygrad = function_sol_grad(xvals)
    return np.reshape(xvals, (-1, 1)), np.reshape(yvals, (-1, 1)), np.reshape(ygrad, (-1, 1))


if __name__ == '__main__':

    config = get_config()

    # Create directories if not exist.
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)

    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)

    sys.stdout = visual_data.Logger(
        os.path.join(
            config.log_dir,
            'train-{}-{}.log'.format(
                "NN" if config.net_type == 'pinn' else "gNN, w={}".format(config.g_weight),
                config.num_train_points
            )
        ),
        sys.stdout
    )

    print(config)

    # Train data.
#     train_x, train_u, _ = gen_all(config.num_train_points)
#     valid_x, valid_u, valid_u_g = gen_all(config.num_test_points)

#     train_data = np.stack((train_x, train_u))

    # Model
    # PINNs/gPINNs
    layers = [1] + [20] * 3 + [1]
    net_NN = FCNet(layers)
    model_NN = FunctionPINNs(net_NN)

    if config.net_type == 'gpinn':
        model_NN.w_g = config.g_weight

    # solver_NN = Solver(train_data, model_NN, config)
    # solver_NN.train()

    # Plot
    # plt.rcParams.update({"font.size": 16})
    #
    # # Exact u
    # plt.figure(4, figsize=(20, 15))
    # plt.plot(valid_x, valid_u, label="Exact", color="black")
    #
    # u_pred = solver_NN.inference(valid_x)[0]
    # plt.plot(valid_x, u_pred.detach().numpy(),
    #          label="{}".format("NN" if config.net_type == "pinn" else "gNN, w={}".format(config.g_weight)),
    #          color="red", linestyle="dashed")
    #
    # plt.xlabel("x")
    # plt.ylabel("u")
    # plt.legend(frameon=False)
    # plt.savefig(os.path.join(config.result_dir,
    #                          'figure/function/u-{}-{}.png'.format(config.net_type, config.num_train_points)),
    #             dpi=120)
    #
    # # Exact u`
    # plt.figure(4, figsize=(20, 15))
    # plt.clf()
    # plt.plot(valid_x, valid_u_g, label="Exact", color="black")
    # u_g_pred = solver_NN.inference(valid_x)[1]
    # plt.plot(valid_x, u_g_pred.detach().numpy(),
    #          label="{}".format("NN" if config.net_type == "pinn" else "gNN, w={}".format(config.g_weight)),
    #          color="red", linestyle="dashed")
    #
    # plt.xlabel("x")
    # plt.ylabel("u`")
    # plt.legend(frameon=False)
    # plt.savefig('./result/figure/function/u_g-{}-{}.png'.format(config.net_type, config.num_train_points), dpi=120)

    # L2 of u/u`
    # u_pred = solver_NN.inference(valid_x)[0]
    # u_g_pred = solver_NN.inference(valid_x)[1]
    #
    # l2_u = ((valid_u - u_pred.detach().numpy())**2).mean()
    # l2_u_g = ((valid_u_g - u_g_pred.detach().numpy())**2).mean()

    L2_u = {}
    L2_u_g = {}

    for i in range(5, 31):
        
        train_x, train_u, _ = gen_all(config.num_train_points)
        valid_x, valid_u, valid_u_g = gen_all(config.num_test_points)

        train_data = np.stack((train_x, train_u))
        
        config.num_train_points = int(i)

        solver_NN = Solver(train_data, model_NN, config)

        # Train
        solver_NN.train()

        # Predict
        u_pred = solver_NN.inference(valid_x)[0]
        u_g_pred = solver_NN.inference(valid_x)[1]

        l2_u = ((valid_u - u_pred.detach().numpy())**2).mean()
        l2_u_g = ((valid_u_g - u_g_pred.detach().numpy())**2).mean()

        # Save loss dict
        L2_u['training point-{}'.format(config.num_train_points)] = l2_u
        L2_u_g['training point-{}'.format(config.num_train_points)] = l2_u_g

    # Save L2 relative error of u/u`
    save_dir = os.path.join(config.result_dir, 'loss/function')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    n = random.randint(1, 10)
    torch.save(L2_u, os.path.join(save_dir, 'l2_u-{}-{}'.format(config.net_type, n)))
    torch.save(L2_u_g, os.path.join(save_dir, 'l2_u_g-{}-{}'.format(config.net_type, n)))



