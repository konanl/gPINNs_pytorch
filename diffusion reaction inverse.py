#!/usr/bin/env python
# encoding: utf-8
"""
    @author: LiangL. Yan
    @license: (C) Copyright forever
    @contact: yokingyan@gmail.com
    @ide: pycharm
    @file: diffusion reaction inverse.py
    @time: 2022/10/15 15:25
    @desc:
    
"""
import argparse
import matplotlib.pyplot as plt
from solver import *
from model import *
from process_data import *
from scipy.integrate import solve_bvp
import visual_data
import sys


def get_config():
    parser = argparse.ArgumentParser("PINNs/gPINNs for forward diffusion reaction model", add_help=False)

    # Training configurations.
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--net_type', default='pinn', type=str)
    parser.add_argument('--num_epochs', default=200000, type=int)
    parser.add_argument('--resume_epoch', type=int, default=0, help='resume training from this step')
    parser.add_argument('--num_epochs_decay', type=int, default=70000, help='decay lr from this step')
    parser.add_argument('--num_train_points', type=int, default=10, help='the number of training points')
    parser.add_argument('--g_weight', type=float, default=0.01)
    parser.add_argument('--output_transform', type=bool, default=True)
    parser.add_argument('--num_supervised_train_points', type=int, default=8,
                        help='the number of supervised training points')
    # Testing configurations.
    parser.add_argument('--test_epochs', type=int, default=80000, help='how long we should test model')
    parser.add_argument('--num_test_points', type=int, default=10000, help='the number of testing points')
    # Directories.
    parser.add_argument('--model_save_dir', type=str, default='./models/diffusion-reaction-inverse/model')
    parser.add_argument('--result_dir', type=str, default='./result')
    parser.add_argument('--log_dir', type=str, default='./models/diffusion-reaction-inverse/logs')
    # Step size.
    parser.add_argument('--log_step', type=int, default=50000)
    parser.add_argument('--model_save_step', type=int, default=50000)
    parser.add_argument('--lr_update_step', type=int, default=10000)

    return parser.parse_args()


l = 0.01


def k(x):
    return 0.1 + np.exp(-0.5 * (x - 0.5) ** 2 / 0.15 ** 2)


def fun(x, y):
    return np.vstack((y[1], 100 * (k(x) * y[0] + np.sin(2 * np.pi * x))))


def bc(ya, yb):
    return np.array([ya[0], yb[0]])


a = np.linspace(0, 1, 1000)
b = np.zeros((2, a.size))

res = solve_bvp(fun, bc, a, b)


def sol(x):
    return res.sol(x)[0]


def du(x):
    return res.sol(x)[1]


class DiffusionReactionInverse(PINNs):
    def __init__(self, net_, w=[1, 0], optimizer_=optim):
        super(DiffusionReactionInverse, self).__init__(net_, w=w, optimizer_=optimizer_)

        self.params_true = k_true

    def output_transform(self, x, y):
        return torch.concat((torch.tanh(x) * torch.tanh(1 - x) * y[:, 0:1], y[:, 1:2]), dim=1)

    def k(self, x):
        return 0.1 + torch.exp(-0.5 * (x - 0.5) ** 2 / 0.15 ** 2)

    def pde(self, x):
        y = self.forward(x)
        y = self.output_transform(x, y)

        u = y[:, 0:1]
        k = y[:, 1:2]

        dudx = gradients(x, u)
        d2udx2 = gradients(x, dudx)
        eqs = l * d2udx2 - k * u - torch.sin(2 * np.pi * x)
        g_eqs = torch.tensor(0.0, dtype=torch.float32)

        if self.w_g:
            d3udx3 = gradients(x, d2udx2)
            dkdx = gradients(x, k)
            g_eqs = l * d3udx3 - k * dudx - u * dkdx - 2 * np.pi * torch.cos(2 * np.pi * x)

        return [eqs, g_eqs]

    def get_params(self, x):
        return self.forward(x)[:, 1:2]


def gen_traindata(num):
    xvals = np.linspace(0, 1, num)
    yvals = sol(xvals)

    return np.reshape(xvals, (-1, 1)), np.reshape(yvals, (-1, 1))


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

    # Data
    # supervised data
    ob_x, ob_u = gen_traindata(config.num_supervised_train_points)

    # unsupervised data
    data = Data([8, 2], [0, 1], "uniform", nums_test=1000)
    train_x, train_u = data.train_data, sol(data.train_data)
    valid_x, valid_u = gen_traindata(config.num_test_points)
    k_true = k(train_x)

    train_data = np.stack((train_x, train_u))
    observe_data = np.stack((ob_x, ob_u))

    # Model
    # PINNs/gPINNs
    layers = [1] + [20] * 3 + [2]
    # There are two output of network, one is to predict u, another is to predict k
    net_NN = FCNet(layers)
    model_NN = DiffusionReactionInverse(net_NN, w=[1, 0], optimizer_="Inverse")

    if config.net_type == 'gpinn':
        model_NN.w_g = config.g_weight

    solver_NN = Solver(train_data, model_NN, config, observe_data, model_name="diffusion-reaction-inverse")
    solver_NN.train()

    # Plot
    # U
    plt.rcParams.update({"font.size": 20})

    plt.figure(4, figsize=(20, 15))
    plt.plot(valid_x, sol(valid_x), label="Exact", color="black")
    plt.plot(valid_x, solver_NN.inference(valid_x)[0][:, 0:1].detach().numpy(),
             label="{}".format("NN" if config.net_type == "pinn" else "gNN, w={}".format(config.g_weight)),
             color='red',  linestyle="dashed")

    plt.xlabel("x")
    plt.ylabel("u")
    plt.legend(frameon=False)
    plt.savefig(os.path.join(config.result_dir,
                             'figure/diffusion-reaction-inverse/u-{}-{}.png'
                             .format(config.net_type, config.num_train_points)),
                dpi=120)

    # K
    plt.clf()
    plt.figure(4, figsize=(20, 15))
    plt.plot(valid_x, k(valid_x), label="Exact", color="black")
    k_pred = solver_NN.model(torch.tensor(valid_x, dtype=torch.float32))[:, 1:2]
    plt.plot(valid_x, k_pred.detach().numpy(),
             label="{}".format("NN" if config.net_type == "pinn" else "gNN, w={}".format(config.g_weight)),
             color='red', linestyle="dashed")
    plt.xlabel("x")
    plt.ylabel("K")
    plt.legend(frameon=False)
    plt.savefig(os.path.join(config.result_dir,
                             'figure/diffusion-reaction-inverse/k-{}-{}.png'
                             .format(config.net_type, config.num_train_points)),
                dpi=120)

    # U`
    plt.clf()
    plt.figure(4, figsize=(20, 15))
    plt.plot(valid_x, du(valid_x), label="Exact", color="black")
    u_g_pred = solver_NN.inference(valid_x)[1][:, 0:1].detach().numpy()
    plt.plot(valid_x, u_g_pred,
             label="{}".format("NN" if config.net_type == "pinn" else "gNN, w={}".format(config.g_weight)),
             color='red', linestyle="dashed")
    plt.xlabel("x")
    plt.ylabel("u`")
    plt.legend(frameon=False)
    plt.savefig(os.path.join(config.result_dir,
                             'figure/diffusion-reaction-inverse/u`-{}-{}.png'
                             .format(config.net_type, config.num_train_points)),
                dpi=120)

    plt.show()






