#!/usr/bin/env python
# encoding: utf-8
"""
    @author: LiangL. Yan
    @license: (C) Copyright forever
    @contact: yokingyan@gmail.com
    @ide: pycharm
    @file: burger.py
    @time: 2022/10/18 16:49
    @desc:
    
"""
# Forward Question

import argparse
import time
import datetime

import numpy as np
import torch
from tqdm import tqdm
from SALib.sample import sobol_sequence
import matplotlib.pyplot as plt
import os
from model import *
from process_data import *
import visual_data
import sys


def get_config():
    parser = argparse.ArgumentParser("PINNs/gPINNs for burger", add_help=False)

    # Training configurations.
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--net_type', default='pinn', type=str)
    parser.add_argument('--train_model', default='none', type=str)
    parser.add_argument('--num_epochs', default=20000, type=int)
    parser.add_argument('--resume_epoch', type=int, default=0, help='resume training from this step')
    parser.add_argument('--num_epochs_decay', type=int, default=17000, help='decay lr from this step')
    parser.add_argument('--num_train_points', type=int, default=1500, help='the number of training points')
    parser.add_argument('--g_weight', type=float, default=0.0001)
    parser.add_argument('--output_transform', type=bool, default=True)

    # Testing configurations.
    parser.add_argument('--test_epochs', type=int, default=8000, help='how long we should test model')
    parser.add_argument('--num_test_points', type=int, default=1000, help='the number of testing points')
    # Directories.
    parser.add_argument('--model_save_dir', type=str, default='./models/burger/model')
    parser.add_argument('--result_dir', type=str, default='./result')
    parser.add_argument('--log_dir', type=str, default='./models/burger/logs')
    # Step size.
    parser.add_argument('--log_step', type=int, default=2000)
    parser.add_argument('--model_save_step', type=int, default=5000)
    parser.add_argument('--lr_update_step', type=int, default=10000)

    return parser.parse_args()


def gen_testdata():
    data = np.load("./data/Burgers.npz")
    t, x, exact = data["t"], data["x"], data["usol"].T
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = exact.flatten()[:, None]
    return X.astype(np.float32), y.astype(np.float32)


def gen_traindata(N, method='uniform'):

    if method == 'uniform':
        Nt = int((N / 2) ** 0.5)
        Nx = 2 * Nt
        x = np.linspace(-1, 1, Nx, endpoint=False)
        t = np.linspace(0, 1, Nt, endpoint=False)
        xx, tt = np.meshgrid(x, t)
    elif method == 'sobol':
        # n = int(N*0.05)
        a = sobol_sequence.sample(N, 2)
        xx = a[:, 0:1] * 2 - 1
        tt = a[:, 1:2]
    else:
        n = int(N * 0.05)
        xx = np.random.random(N) * 2 - 1
        tt = np.random.random(N)

    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    return X.astype(np.float32)


# PINN + RAR
class BurgerPINNs(PINNs):
    def __init__(self, net_, w=[1, 0]):
        super(BurgerPINNs, self).__init__(net_, w=w)

    def output_transform(self, x, y):
        """Output transform."""
        x_in = x[:, 0:1]
        t_in = x[:, 1:2]

        return (1 - x_in) * (1 + x_in) * (1 - torch.exp(-t_in)) * y - torch.sin(np.pi * x_in)

    def pde(self, x):
        y = self.forward(x)
        y = self.output_transform(x, y)

        dy = gradients(x, y)
        d2y = gradients(x, dy)

        dydx = dy[:, 0:1]
        dydt = dy[:, 1:2]
        d2ydx2 = d2y[:, 0:1]

        eqs = dydt + y * dydx - 0.01 / np.pi * d2ydx2

        if self.w_g:
            # d2ydxt = gradients(x, dydx)[:, 1:2]
            d2ydtx = gradients(x, dydt)[:, 0:1]
            d3ydx3 = gradients(x, d2ydx2)[:, 0:1]
            d2ydt2 = d2y[:, 1:2]
            d3ydx2t = gradients(x, d2ydx2)[:, 1:2]

            g_eqs_x = d2ydtx + (dydx * dydx + y * d2ydx2) - 0.01 / np.pi * d3ydx3
            g_eqs_t = d2ydt2 + dydt * dydx + y * d2ydtx - 0.01 / np.pi * d3ydx2t

        else:
            g_eqs_x = torch.zeros((1,), dtype=torch.float32)
            g_eqs_t = torch.zeros((1,), dtype=torch.float32)

        return [eqs, g_eqs_x, g_eqs_t]


def l2_relative_error(y_true, y_pred):
    """L2 norm relative error."""
    if isinstance(y_pred, np.ndarray):
        return np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true)
    else:
        y_pred = np.array(y_pred.detach().numpy())
        return np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true)


def restore_model(model, resume_epochs, config):
    """Restore the trained PINNs/gPINNs."""
    print('Loading the trained models from step {}...'.format(resume_epochs))
    if config.net_type == 'gpinn':
        path = os.path.join(config.model_save_dir,
                            '{}-gPINNs-w_g-{}.ckpt'.format(resume_epochs, str(config.g_weight)))
    elif config.net_type == 'pinn':
        path = os.path.join(config.model_save_dir, '{}-PINNs.ckpt'.format(resume_epochs))

    model.net.load_state_dict(torch.load(path))
    print("Success load model with epoch {} !!!".format(resume_epochs))


def train(data, model, optimizer, config):
    """Train PINNs/gPINNs with RAR"""

    # Train data.
    train_data = torch.tensor(data, dtype=torch.float32, requires_grad=True)

    lr = config.lr

    start_epoch = 0
    if config.resume_epoch:
        start_epoch = config.resume_epoch
        restore_model(model, start_epoch, config)

    # Start train.
    if config.net_type == 'pinn':
        print('\nStrat training PINNs...\n')
    else:
        print('\nStart training gPINNs w_g={}\n'.format(str(config.g_weight)))

    start_time = time.time()
    with tqdm(range(start_epoch, config.num_epochs)) as tepochs:
        for epoch in tepochs:

            # Set the begin
            tepochs.set_description(f"Epoch {epoch + 1}")

            # Compute loss
            loss_sum = 0
            y_pred = model(train_data)
            y_pred = model.output_transform(train_data, y_pred)
            loss = model.pde(train_data)
            # EQLoss = model.w_f * torch.mean(torch.square(loss[0])) + \
            #          model.w_g * torch.mean(torch.square(loss[1])) + model.w_g * torch.mean(torch.square(loss[2]))
            EqLoss = model.w_f * torch.mean(torch.square(loss[0]))
            gEqLoss = model.w_g * torch.mean(torch.square(loss[1])) + model.w_g * torch.mean(torch.square(loss[2]))

            Loss = EqLoss + gEqLoss

            # Backward and optimize.
            Loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Logging
            loss = {}
            loss_sum += Loss
            if config.net_type == "pinn":
                loss['PINNs/loss'] = Loss.item()
                loss['PINNs/EqLoss'] = EqLoss.item()
                loss['PINNs/gEqLoss'] = gEqLoss.item()
            else:
                loss['gPINNs, w={}/loss'.format(config.g_weight)] = Loss.item()
                loss['gPINNs, w={}/EqLoss'.format(config.g_weight)] = EqLoss.item()
                loss['gPINNs, w={}/gEqLoss'.format(config.g_weight)] = gEqLoss.item()

            # Save loss information.
            if (epoch + 1) % config.log_step == 0:
                loss_save_path = os.path.join(
                    config.log_dir,
                    "{}-{}-{}".format(
                        "NN" if config.net_type == 'pinn' else "gNN-w_g_{}".format(config.g_weight),
                        config.num_train_points,
                        epoch + 1
                    )
                )
                torch.save(loss, loss_save_path)

            # Print out training information.
            if (epoch + 1) % config.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, epoch + 1, config.num_epochs)
                for tag, value in loss.items():
                    log += ", {}: {:.2e}".format(tag, value)
                print(log)

            # Print out training information.
            if (epoch + 1) % config.model_save_step == 0:
                if config.net_type == 'pinn':
                    path = os.path.join(config.model_save_dir, '{}-PINNs.ckpt'.format(epoch + 1))
                else:
                    path = os.path.join(config.model_save_dir,
                                        '{}-gPINNs-w_g-{}.ckpt'.format(epoch + 1, config.g_weight))
                torch.save(model.net.state_dict(), path)
                print('Saved model checkpoints into {}...'.format(config.model_save_dir))

            # Decay learning rates.
            if (epoch + 1) % config.lr_update_step == 0 and (epoch + 1) > (config.num_epochs - config.num_epochs_decay):
                lr -= (config.lr / float(config.num_epochs_decay))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                print('Decayed learning rates, lr: {}.'.format(lr))

            MLoss = torch.mean(loss_sum)
            tepochs.set_postfix(Meanloss=MLoss.item())
            time.sleep(0.0001)

    # Save parameters information.


def predict(model, X, config):
    """Predicting trained model."""
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32, requires_grad=True)

    # Loading trained model
    restore_model(model, config.num_epochs, config)
    y_pred = model(X)
    y_pred = model.output_transform(X, y_pred)
    return y_pred


def plot_scatter(add_points, num):
    add_points = np.reshape(add_points, (-1, 2))
    x = add_points[:, 0:1]
    t = add_points[:, 1:2]
    plt.clf()
    plt.rcParams['font.size'] = 20
    plt.figure(2, figsize=(10, 8))
    plt.scatter(x, t, s=0.8, color="red")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.savefig("./result/figure/burger/figure_11_A-{}-gpinn.png".format(num), bbox_inches='tight')


def plot_colormap(x_true, error_u, error_pde, num):
    plt.clf()
    plt.rcParams['font.size'] = 20
    plt.figure(2, figsize=(10, 8))
    plt.pcolormesh(x_true[..., 0], x_true[..., 1], error_u[..., 0], cmap='viridis',
                   shading='gouraud', antialiased=True, snap=True
                   )
    plt.xlabel("x")
    plt.ylabel("t")
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=20)
    plt.tight_layout()
    plt.title("Error of $u$")
    plt.savefig("./result/figure/burger/figure_11_B-{}-gpinn.png".format(num), bbox_inches='tight')

    plt.clf()
    plt.figure(2, figsize=(10, 8))
    plt.pcolormesh(x_true[..., 0], x_true[..., 1], error_pde[..., 0].detach().numpy(), cmap='viridis',
                   shading='gouraud', antialiased=True, snap=True
                   )
    plt.xlabel("x")
    plt.ylabel("t")
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=20)
    plt.tight_layout()
    plt.title("Error of PDE residual")
    plt.savefig("./result/figure/burger/figure_11_C-{}-gpinn.png".format(num), bbox_inches='tight')


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

    # Train data.
    valid_x, valid_u = gen_testdata()
    train_x = gen_traindata(config.num_train_points, method='sobol')

    # Model
    # PINNs/gPINNs
    layers = [2] + [32] * 3 + [1]
    net_NN = FCNet(layers)
    model_NN = BurgerPINNs(net_NN)

    if config.net_type == 'gpinn':
        model_NN.w_g = config.g_weight

    train_model = config.train_model

    # Optimizer.
    optimizer = torch.optim.Adam(params=model_NN.parameters(), lr=config.lr, betas=(0.9, 0.999))

    # Train without RAR
    # train(train_x, model_NN, optimizer, config)

    def train_non_RAR(model=False):
        if model:
            training_points = [1500, 2000, 2500, 3000]

            # Figure 10.
            L2_relative_error = {}
            for training_point in training_points:

                print("\n###########################################")
                print("# Start training with training_point:{} #".format(training_point))
                print("###########################################")

                config.num_train_points = int(training_point)

                data = gen_traindata(config.num_train_points, method='random')

                train(data, model_NN, optimizer, config)

                u_pred = predict(model_NN, valid_x, config)

                print("L2 relative error:", l2_relative_error(valid_u, u_pred))

                L2_relative_error[training_point] = l2_relative_error(valid_u, u_pred)

            # Save L2 relative error
            save_dir = os.path.join(config.result_dir, 'loss/burger')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            torch.save(L2_relative_error, os.path.join(save_dir, 'l2_error-{}'.format(
                'pinn' if config.net_type == "pinn" else "gpinn-w_{}".format(config.g_weight))))

    def train_RAR(data, model, config, train_model=False):
        if train_model:
            L2_error = {}
            add_points = []
            for i in range(40):
                X = torch.tensor(gen_traindata(100000, method='sobol'), dtype=torch.float32, requires_grad=True)
                err_eq = np.abs(model.pde(X)[0].detach().numpy())
                err = np.mean(err_eq)
                print("Mean residual: %.3e" % err)
                err_eq = torch.tensor(err_eq)
                x_ids = torch.topk(err_eq, 10, dim=0)[1].numpy()

                for elem in x_ids:
                    print("Adding new point:", X[elem], "\n")
                    add_points.append(X[elem].detach().numpy().tolist()[0])
                    data.tolist()
                    data = np.vstack((data, X[elem].detach().numpy()))

                optimizer_ = torch.optim.Adam(params=model_NN.parameters(), lr=config.lr, betas=(0.9, 0.999))

                print("\n##### Train with training points:{} #####\n".format(len(data)))

                train(data, model, optimizer_, config)

                valid_x_, valid_u_ = gen_testdata()
                x_true = valid_x_.reshape((100, 256, 2))
                u_true = valid_u_.reshape((100, 256, 1))
                u_pred = predict(model, x_true, config)
                error_u = abs(u_true - u_pred.detach().numpy())
                error_pde = abs(model_NN.pde(torch.tensor(valid_x_, requires_grad=True,
                                                          dtype=torch.float32))[0].reshape((100, 256, 1)))
                print("L2 relative error:", l2_relative_error(u_true, u_pred))

                L2_error[int(len(data))] = l2_relative_error(u_true, u_pred)

                # Plot Residual points when every 100 added
                print("The numbers of added points: {}\n".format(len(add_points)))
                if len(add_points) % 100 == 0:
                    plot_scatter(add_points, len(data))
                    plot_colormap(x_true, error_u, error_pde, len(data))

            save_dir = os.path.join(config.result_dir, 'loss/burger')
            torch.save(L2_error, os.path.join(save_dir, 'l2_error-{}_RAR'.format(
                'pinn' if config.net_type == "pinn" else "gpinn-w_{}".format(config.g_weight))))


    if config.train_model == "none RAR":
        train_non_RAR(True)
    elif config.train_model == "RAR":
        train_non_RAR(False)
        train_RAR(train_x, model_NN, config, True)
    else:
        print("No choose train model !!!")

    # Plot figure.11 A,B,C
    #############################################################
    x_true = valid_x.reshape((100, 256, 2))
    u_true = valid_u.reshape((100, 256, 1))
    u_pred = predict(model_NN, x_true, config)

    error_u = abs(u_pred.detach().numpy() - u_true)
    plt.rcParams['font.size'] = 20
    plt.figure(2, figsize=(10, 8))
    plt.pcolormesh(x_true[..., 0], x_true[..., 1], error_u[..., 0], cmap='viridis',
                   shading='gouraud', antialiased=True, snap=True
                   )
    plt.xlabel("x")
    plt.ylabel("t")
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=20)
    plt.tight_layout()
    plt.title("Error of $u$")
    plt.savefig("./result/figure/burger/figure_11_B-gpinn.png")

    plt.clf()
    plt.figure(2, figsize=(10, 8))
    error_pde = abs(model_NN.pde(torch.tensor(valid_x, requires_grad=True,
                                              dtype=torch.float32))[0].reshape((100, 256, 1)))
    plt.pcolormesh(x_true[..., 0], x_true[..., 1], error_pde[..., 0].detach().numpy(), cmap='viridis',
                   shading='gouraud', antialiased=True, snap=True
                   )
    plt.xlabel("x")
    plt.ylabel("t")
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=20)
    plt.tight_layout()
    plt.title("Error of PDE residual")
    plt.savefig("./result/figure/burger/figure_11_C-gpinn.png")

    plt.show()
    #############################################################


