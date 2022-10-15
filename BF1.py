#!/usr/bin/env python
# encoding: utf-8
"""
    @author: LiangL. Yan
    @license: (C) Copyright forever
    @contact: yokingyan@gmail.com
    @ide: pycharm
    @file: BF1.py
    @time: 2022/10/4 20:28
    @desc:
    
"""
import argparse

from visualization import *
from pde import *
from solver import *
from model import *
from process_data import *
import visual_data


def get_config():
    parser = argparse.ArgumentParser("PINNs for Brinkman-Forchheimer model case 1", add_help=False)

    # Training configurations.
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--net_type', default='pinn', type=str)
    parser.add_argument('--num_epochs', default=50000, type=int)
    parser.add_argument('--resume_epoch', type=int, default=0, help='resume training from this step')
    parser.add_argument('--num_epochs_decay', type=int, default=25000, help='decay lr from this step')
    parser.add_argument('--num_train_points', type=int, default=10, help='the number of training points')
    parser.add_argument('--g_weight', type=float, default=0.1)
    parser.add_argument('--num_supervised_train_points', type=int, default=5,
                        help='the number of supervised training points')
    parser.add_argument('--output_transform', type=bool, default=True)
    # Testing configurations.
    parser.add_argument('--test_epochs', type=int, default=40000, help='how long we should test model')
    parser.add_argument('--num_test_points', type=int, default=1000, help='the number of testing points')
    # Directories.
    parser.add_argument('--model_save_dir', type=str, default='./models/BF/case 1/model')
    parser.add_argument('--result_dir', type=str, default='./result')
    parser.add_argument('--log_dir', type=str, default='./models/BF/case 1/logs')
    # Step size.
    parser.add_argument('--log_step', type=int, default=5000)
    parser.add_argument('--model_save_step', type=int, default=10000)
    parser.add_argument('--lr_update_step', type=int, default=10000)

    return parser.parse_args()


class BFPINNs(PINNs):
    """PINNs model for BF inverse model"""
    def __init__(self, net_, w=[1, 0], optimizer_=optim, v_e=1e-3):
        super(BFPINNs, self).__init__(net_, w=w, optimizer_=optimizer_)

        # the parameter which need to predict in PINNs inverse problem
        self.v_e_ = nn.Parameter(torch.tensor([0.1, ]))
        self.params_true = [v_e, ]

    def output_transform(self, x, y):
        return torch.tanh(x) * torch.tanh(1 - x) * y

    def pde(self, x):
        y = self.forward(x)
        y = self.output_transform(x, y)

        dydx = gradients(x, y)
        d2ydx2 = gradients(x, dydx)
        v_e = torch.log(torch.exp(self.v_e_) + 1) * 0.1
        eqs = -v_e/e * d2ydx2 + v * y/K - g
        g_eqs = torch.tensor(0.0, dtype=torch.float32)

        if self.w_g:
            d3udx3 = gradients(x, d2ydx2)
            g_eqs = -v_e/e * d3udx3 + v/K * dydx

        return [eqs, g_eqs]

    def get_params(self):
        return [torch.log(torch.exp(self.v_e_) + 1) * 0.1]


def gen_traindata(num):
    xvals = np.linspace(1 / (num + 1), 1, num, endpoint=False, dtype=np.float32)
    yvals = BF_sol(xvals)
    dy = BF_grad(xvals)

    return np.reshape(xvals, (-1, 1)), np.reshape(yvals, (-1, 1)), dy


def gen_observedata(num):
    if num:
        xvals = np.linspace(1 / num, 1 - 1 / num, num, endpoint=False, dtype=np.float32)
        yvals = BF_sol(xvals)

        return np.reshape(xvals, (-1, 1)), np.reshape(yvals, (-1, 1))
    else:
        return 0


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
    # train_x, train_u, _= gen_traindata(config.num_train_points)
    # valid_x, valid_u, _ = gen_traindata(config.num_test_points)

    # train_data = np.stack((train_x, train_u))
    # observe_data = np.stack((ob_x, ob_u))

    # Model
    # PINNs/gPINNs
    layers = [1] + [20] * 3 + [1]
    net_NN = FCNet(layers)
    model_NN = BFPINNs(net_NN, w=[1, 0], optimizer_="Inverse")

    if config.net_type == 'gpinn':
        model_NN.w_g = config.g_weight

    # print("before train {} ".format(model_NN.get_params()))
    # solver_NN = Solver(train_data, model_NN, config, observe_data, model_name="BF/case_1")
    # solver_NN.train()
    # print("after train {} ".format(model_NN.get_params()))

    # Plot
    # plt.rcParams.update({"font.size": 20})
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
    #                          'figure/BF/case 1/u-{}-{}.png'.format(config.net_type, config.num_train_points)),
    #             dpi=120)

    # Figure.6 A,B,C
    Error_v_e = {}
    L2_u = {}
    L2_dudx = {}

    training_points = [5, 10, 15, 20, 25, 30, 35]

    for training_point in training_points:

        training_point = int(training_point)

        config.num_supervised_train_points = training_point

        train_x, train_u, _ = gen_traindata(config.num_train_points)
        ob_x, ob_u = gen_observedata(config.num_supervised_train_points)

        valid_x, valid_u, valid_u_g = gen_traindata(config.num_test_points)

        train_data = np.stack((train_x, train_u))
        observe_data = np.stack((ob_x, ob_u))

        solver_NN = Solver(train_data, model_NN, config, observe_data, model_name="BF/case_1")

        # Train
        solver_NN.train()

        # Predict
        v_e_pred = solver_NN.getModelParams()[0]
        v_e_loss = ((v_e_pred - 0.001)**2)

        u_pred = solver_NN.inference(valid_x)[0]
        l2_u = ((valid_u - u_pred.detach().numpy()) ** 2).mean()

        dudx_pred = solver_NN.inference(valid_x)[1]
        l2_dudx = (valid_u_g - dudx_pred.detach().numpy()).mean()

        # Save loss dict
        Error_v_e['training point-{}'.format(config.num_supervised_train_points)] = v_e_loss
        L2_u['training point-{}'.format(config.num_supervised_train_points)] = l2_u
        L2_dudx['training point-{}'.format(config.num_supervised_train_points)] = l2_dudx

    # Save L2 relative error of u/u` and PDE residual
    save_dir = os.path.join(config.result_dir, 'loss/BF/case_1')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    n = random.randint(1, 10)
    torch.save(L2_u, os.path.join(save_dir, 'l2_u-{}-{}'.format(
        'pinn' if config.net_type == "pinn" else "gpinn-w_{}".format(config.g_weight), n)))
    torch.save(L2_dudx, os.path.join(save_dir, 'l2_u_g-{}-{}'.format(
        'pinn' if config.net_type == "pinn" else "gpinn-w_{}".format(config.g_weight), n)))
    torch.save(Error_v_e, os.path.join(save_dir, 'error_v_e-{}-{}'.format(
        'pinn' if config.net_type == "pinn" else "gpinn-w_{}".format(config.g_weight), n)))




