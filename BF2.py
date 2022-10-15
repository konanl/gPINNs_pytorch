#!/usr/bin/env python
# encoding: utf-8
"""
    @author: LiangL. Yan
    @license: (C) Copyright forever
    @contact: yokingyan@gmail.com
    @ide: pycharm
    @file: BF2.py
    @time: 2022/10/14 09:27
    @desc:
    
"""
import argparse

from visualization import *
from solver import *
from model import *
from process_data import *
import visual_data

g = 1
v = 1e-3
e = 0.4
H = 1


def get_config():
    parser = argparse.ArgumentParser("PINNs for Brinkman-Forchheimer model case_2", add_help=False)

    # Training configurations.
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--net_type', default='pinn', type=str)
    parser.add_argument('--num_epochs', default=50000, type=int)
    parser.add_argument('--resume_epoch', type=int, default=0, help='resume training from this step')
    parser.add_argument('--num_epochs_decay', type=int, default=40000, help='decay lr from this step')
    parser.add_argument('--num_train_points', type=int, default=10, help='the number of training points')
    parser.add_argument('--g_weight', type=float, default=0.1)
    parser.add_argument('--num_supervised_train_points', type=int, default=5,
                        help='the number of supervised training points')
    parser.add_argument('--output_transform', type=bool, default=True)
    # Testing configurations.
    parser.add_argument('--test_epochs', type=int, default=40000, help='how long we should test model')
    parser.add_argument('--num_test_points', type=int, default=1000, help='the number of testing points')
    # Directories.
    parser.add_argument('--model_save_dir', type=str, default='./models/BF/case_2/model')
    parser.add_argument('--result_dir', type=str, default='./result')
    parser.add_argument('--log_dir', type=str, default='./models/BF/case_2/logs')
    # Step size.
    parser.add_argument('--log_step', type=int, default=5000)
    parser.add_argument('--model_save_step', type=int, default=10000)
    parser.add_argument('--lr_update_step', type=int, default=10000)

    return parser.parse_args()


def sol(x):
    r = (v * e / (1e-3 * 1e-3)) ** 0.5
    return g * 1e-3 / v * (1 - np.cosh(r * (x - H / 2)) / np.cosh(r * H / 2))


def gen_traindata(num):
    xvals = np.linspace(1 / (num + 1), 1, num, dtype=np.float32, endpoint=False)
    yvals = sol(xvals)

    return np.reshape(xvals, (-1, 1)), np.reshape(yvals, (-1, 1))


def gen_observedata(num):
    if num:
        xvals = np.linspace(1 / num, 1 - 1 / num, num, endpoint=False, dtype=np.float32)
        yvals = sol(xvals)

        return np.reshape(xvals, (-1, 1)), np.reshape(yvals, (-1, 1))
    else:
        return 0


def du(x):
    r = (v * e / (1e-3 * 1e-3)) ** 0.5
    return g * 1e-3 / v * (-r * np.sinh(r * (x - H / 2)) / np.cosh(r * H / 2))


class BF2PINNs(PINNs):
    def __init__(self, net_, w=[1, 0], optimizer_=optim, v_e_true=1e-3, k_true=1e-3):
        super(BF2PINNs, self).__init__(net_, w=w, optimizer_=optimizer_)

        # the parameter which need to predict in PINNs inverse problem
        self.v_e_ = nn.Parameter(torch.tensor([0.0, ]))
        self.K_ = nn.Parameter(torch.tensor([0.0, ]))

        self.params_true = [v_e_true, k_true]

    def output_transform(self, x, y):
        return torch.tanh(x) * torch.tanh(1 - x) * y

    def pde(self, x):
        y = self.forward(x)
        y = self.output_transform(x, y)

        dydx = gradients(x, y)
        d2ydx2 = gradients(x, dydx)

        [v_e, K] = self.get_params()
        eqs = -v_e/e * d2ydx2 + v * y/K - g
        g_eqs = torch.tensor(0.0, dtype=torch.float32)

        if self.w_g:
            d3ydx3 = gradients(x, d2ydx2)
            g_eqs = -v_e/e * d3ydx3 + v/K * dydx

        return [eqs, g_eqs]

    def get_params(self):
        return [torch.log(torch.exp(self.v_e_) + 1) * 0.1, torch.log(torch.exp(self.K_) + 1) * 0.1]


if __name__ == "__main__":

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
    ob_x, ob_u = gen_observedata(config.num_supervised_train_points)

    # unsupervised data
    train_x, train_u = gen_traindata(config.num_train_points)
    valid_x, valid_u = gen_traindata(config.num_test_points)

    train_data = np.stack((train_x, train_u))
    observe_data = np.stack((ob_x, ob_u))

    # add Gaussian noise
    sigma = 0.05
    noise = np.random.normal(0, sigma, (config.num_supervised_train_points, 1))
    ob_x = ob_x + noise

    # Model
    # PINNs/gPINNs
    layers = [1] + [20] * 3 + [1]
    net_NN = FCNet(layers)
    model_NN = BF2PINNs(net_NN, w=[1, 0], optimizer_="Inverse")

    if config.net_type == 'gpinn':
        model_NN.w_g = config.g_weight

    solver_NN = Solver(train_data, model_NN, config, observe_data, model_name="BF/case_2")
    solver_NN.train()

    # Plot
    # plt.rcParams.update({"font.size": 20})
    #
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
    #                          'figure/BF/case 2/u-{}-{}_add_GS_noise.png'.format(
    #                              "NN" if config.net_type == "pinn" else "gNN, w={}".format(config.g_weight),
    #                              config.num_train_points)
    #                          ),
    #             dpi=120)
