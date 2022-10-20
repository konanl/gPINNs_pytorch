#!/usr/bin/env python
# encoding: utf-8
"""
    @author: LiangL. Yan
    @license: (C) Copyright forever
    @contact: yokingyan@gmail.com
    @ide: pycharm
    @file: solver.py
    @time: 2022/10/3 15:37
    @desc:
    
"""
import time

import torch.nn

from tqdm import tqdm

import torch.optim as optim
import os
from process_data import *
import datetime


class Solver(object):
    """Class to solver PINNs.
    :parameter
        model:
        data:
        config:

    """

    def __init__(self, data_, model, config, ob_data=None, model_name='BF'):
        """Init of Solver."""

        # Data loader.
        self.input_data = data_[0]
        self.output_data = data_[1][:, 0][:, None]

        # Observe data.
        if ob_data is not None:
            self.ob_input_data = ob_data[0]
            self.ob_output_data = ob_data[1][:, 0][:, None]
        else:
            self.ob_input_data = None
            self.ob_output_data = None

        # Basic physics informed neural network.
        self.model = model
        self.model_name = model_name

        # Model configurations.
        self.optimizer = None

        # Training configurations.
        self.net_type = config.net_type
        self.lr = config.lr
        self.num_epochs = config.num_epochs
        self.resume_epoch = config.resume_epoch
        self.num_epochs_decay = config.num_epochs_decay
        self.num_train_points = config.num_train_points
        self.output_transform = config.output_transform
        self.num_supervised_data = config.num_supervised_train_points

        # Testing configurations.
        self.test_epochs = config.test_epochs
        self.num_test_points = config.num_test_points

        # Step size.
        self.log_step = config.log_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Directories.
        self.log_dir = config.log_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Initialize model
        self.build_model()

    def whatPINN(self):
        """Judge what type of PINNs it is."""
        if self.model.w_g != 0:
            return 'gPINNs'
        else:
            return 'PINNs'

    def build_model(self):
        """Constructing model."""
        # 1. Chose optimizer
        if self.model.optimizer == "Adam":
            self.optimizer = optim.Adam(params=self.model.net.parameters(), lr=self.lr, betas=(0.9, 0.999))
        elif self.model.optimizer == "Inverse":
            self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.lr, betas=(0.9, 0.999))
        # 2. Print Information
        if self.whatPINN() == "PINNs":
            print("Success build PINNs !!!")
        else:
            print("Success build gPINNs with w_g = {} !!!".format(self.model.w_g))

    def loss(self, x, y):
        """Loss function of PINNs/gPINNs"""
        loss = self.model.pde(x)
        loss_f = loss[0]
        if self.whatPINN() == 'PINNs':
            return self.model.w_f * torch.mean(torch.square(loss_f))
        else:
            loss_g = loss[1]
            if len(loss) == 3:
                # for loss_ in loss[2:]:
                #     loss_g += self.model.w_g * loss_
                loss_g_ = loss[2]
                return \
                    self.model.w_f * torch.mean(torch.square(loss_f)) + \
                    self.model.w_g * torch.mean(torch.square(loss_g)) + \
                    self.model.w_g * torch.mean(torch.square(loss_g_))

            return self.model.w_f * torch.mean(torch.square(loss_f)) + self.model.w_g * torch.mean(torch.square(loss_g))

    @staticmethod
    def l2_relative_error(y_true, y_pred):
        """L2 norm relative error."""
        if isinstance(y_pred, np.ndarray):
            return np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true)
        else:
            y_pred = np.array(y_pred)
            return np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true)

    def restore_model(self, resume_epochs):
        """Restore the trained PINNs/gPINNs."""
        print('Loading the trained models from step {}...'.format(resume_epochs))
        if self.whatPINN() == 'gPINNs':
            path = os.path.join(self.model_save_dir,
                                '{}-gPINNs-w_g-{}.ckpt'.format(resume_epochs, str(self.model.w_g)))
        elif self.whatPINN() == 'PINNs':
            path = os.path.join(self.model_save_dir, '{}-PINNs.ckpt'.format(resume_epochs))

        self.model.net.load_state_dict(torch.load(path))
        print("Success load model with epoch {} !!!".format(resume_epochs))

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.optimizer.zero_grad()

    def update_lr(self, lr):
        """Decay learning rates of the PINNs/gPINNs."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train_batch(self, **kwargs):
        """Training PINNs/gPINNs of a small batch."""
        pass

    def getModelParams_(self, data):
        """Get model trainable parameters."""
        params = self.model.get_params(data)
        params = params.tolist()
        return [params]

    def getModelParams(self):
        """Get model trainable parameters."""
        params = self.model.get_params()
        if len(params) == 1:
            return [params[0].detach().item()]
        elif len(params) == 2:
            # paramsDict = {'param1': params[0], 'param2': params[1]}
            return [params[0].detach().item(), params[1].detach().item()]
        else:
            params = params.tolist()
            return [params]

    def com_params_loss(self):
        """Compute the loss of parameters."""
        if len(self.model.params_true) == 1:
            return abs(self.model.get_params()[0].detach().numpy() - self.model.params_true[0])
        elif len(self.model.params_true) == 2:
            # 2 params
            param_loss1 = abs(self.model.get_params()[0].detach().numpy() - self.model.params_true[0])
            param_loss2 = abs(self.model.get_params()[1].detach().numpy() - self.model.params_true[1])
            return [param_loss1, param_loss2]

    def train(self, **kwargs):
        # Chose the device

        # device_ = get_default_device()

        # Process train data
        if isinstance(self.input_data, np.ndarray):
            train_data = torch.tensor(self.input_data, requires_grad=True, dtype=torch.float32)
        elif isinstance(self.input_data, torch.Tensor):
            train_data = self.input_data
            train_data.requires_grad = True
        elif isinstance(self.input_data, Dataset):
            pass

        # Learning rate
        lr = self.lr

        # Start train from scratch or resume training
        start_epoch = 0
        if self.resume_epoch:
            start_epoch = self.resume_epoch
            self.restore_model(self.resume_epoch)

        # Start train
        if self.whatPINN() == 'PINNs':
            print('\nStart training PINNs...\n')
        else:
            print('\nStart training gPINNs w_g={}\n'.format(str(self.model.w_g)))

        start_time = time.time()
        params_log = []
        with tqdm(range(start_epoch, self.num_epochs)) as tepochs:
            for epoch in tepochs:

                # Set the begin.
                tepochs.set_description(f"Epoch {epoch + 1}")

                # Compute loss.
                loss_sum = 0
                y_pred = self.model(train_data)
                y_pred = self.model.output_transform(train_data, y_pred)
                EQLoss = self.loss(train_data, y_pred)
                loss_ = self.model.pde(train_data)
                EqLoss = torch.mean(torch.square(loss_[0]))
                DataLoss = 0
                paramsLoss_dri = 0
                if self.ob_input_data is not None:
                    ob_x, ob_y = torch.tensor(self.ob_input_data, requires_grad=True, dtype=torch.float32), \
                                 torch.tensor(self.ob_output_data, requires_grad=True, dtype=torch.float32)
                    y_ob_pred, k_pred = self.model(ob_x)[:, 0:1], self.model(ob_x)[:, 1:2]
                    y_ob_pred = self.model.output_transform(ob_x, y_ob_pred)[:, 0:1]
                    if self.model_name == 'diffusion-reaction-inverse':
                        params_pred = self.getModelParams_(train_data)
                        paramsLoss_dri = torch.mean(torch.abs(self.model.k(ob_x) - k_pred))
                    else:
                        params_pred = self.getModelParams()
                    if len(params_pred) == 1:
                        params_log.append(*params_pred)
                    else:
                        params_log.append(params_pred)
                    paramsLoss = (self.com_params_loss() if self.model_name != 'diffusion-reaction-inverse' \
                        else [paramsLoss_dri.detach().numpy()])
                    DataLoss = ((y_ob_pred - ob_y) ** 2).mean()
                gEqLoss = torch.mean(torch.square(loss_[1]))
                dataLoss = ((y_pred - torch.tensor(self.output_data, dtype=torch.float32)) ** 2).mean()

                if self.model_name == 'diffusion-reaction-inverse':
                    Loss = EQLoss + DataLoss + paramsLoss_dri
                else:
                    Loss = EQLoss + DataLoss

                # Backward and optimize.
                Loss.backward()
                self.optimizer.step()
                self.reset_grad()

                # Logging.
                loss = {}
                loss_sum += Loss
                if self.whatPINN() == 'PINNs':
                    loss['PINNs/loss'] = Loss.item()
                    loss['PINNs/EqLoss'] = EqLoss.item()
                    loss['PINNs/gEqLoss'] = gEqLoss.item()
                    loss['PINNs/dataLoss'] = dataLoss.item()
                    if self.ob_input_data is not None:
                        loss['PINNs/paramLoss_1'] = paramsLoss[0].item()
                        if len(paramsLoss) == 2:
                            loss['PINNs/paramLoss_2'.format(self.model.w_g)] = paramsLoss[1].item()
                else:
                    loss['gPINNs, w={}/loss'.format(self.model.w_g)] = Loss.item()
                    loss['gPINNs, w={}/EqLoss'.format(self.model.w_g)] = EqLoss.item()
                    loss['gPINNs, w={}/gEqLoss'.format(self.model.w_g)] = gEqLoss.item()
                    loss['gPINNs, w={}/dataLoss'.format(self.model.w_g)] = dataLoss.item()
                    if self.ob_input_data is not None:
                        loss['gPINNs, w={}/paramLoss_1'.format(self.model.w_g)] = paramsLoss[0].item()
                        if len(paramsLoss) == 2:
                            loss['gPINNs, w={}/paramLoss_2'.format(self.model.w_g)] = paramsLoss[1].item()

                # Save loss information.
                if (epoch + 1) % self.log_step == 0:
                    loss_save_path = os.path.join(
                        self.log_dir,
                        "{}-{}-{}".format(
                            "NN" if self.net_type == 'pinn' else "gNN-w_g_{}".format(self.model.w_g),
                            self.num_train_points,
                            epoch + 1
                        )
                    )
                    torch.save(loss, loss_save_path)

                # Print out training information.
                if (epoch + 1) % self.log_step == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Elapsed [{}], Iteration [{}/{}]".format(et, epoch + 1, self.num_epochs)
                    for tag, value in loss.items():
                        log += ", {}: {:.2e}".format(tag, value)
                    print(log)

                # Save model checkpoints.
                if (epoch + 1) % self.model_save_step == 0:
                    if self.whatPINN() == 'PINNs':
                        path = os.path.join(self.model_save_dir, '{}-PINNs.ckpt'.format(epoch + 1))
                    else:
                        path = os.path.join(self.model_save_dir,
                                            '{}-gPINNs-w_g-{}.ckpt'.format(epoch + 1, self.model.w_g))
                    torch.save(self.model.net.state_dict(), path)
                    print('Saved model checkpoints into {}...'.format(self.model_save_dir))

                # Decay learning rates.
                if (epoch + 1) % self.lr_update_step == 0 and (epoch + 1) > (self.num_epochs - self.num_epochs_decay):
                    lr -= (self.lr / float(self.num_epochs_decay))
                    self.update_lr(lr)
                    print('Decayed learning rates, lr: {}.'.format(lr))

                MLoss = torch.mean(loss_sum)
                tepochs.set_postfix(Meanloss=MLoss.item())
                time.sleep(0.0001)

            # Save parameters information.
            if len(params_log) != 0:
                params_loss_save_path = os.path.join(
                    self.result_dir,
                    "loss",
                    self.model_name,
                    "param_loss-{}-{}-{}".format(
                        "NN" if self.net_type == 'pinn' else "gNN-w_g_{}".format(self.model.w_g),
                        self.num_train_points,
                        self.num_epochs
                    )
                )
                torch.save(params_log, params_loss_save_path)

    def test(self, **kwargs):
        pass

    def inference(self, X):
        """Predicting trained model."""
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32, requires_grad=True)
            # Loading trained model
            self.restore_model(self.num_epochs)
            y_pred = self.model(X)

            if self.output_transform:
                y_pred = self.model.output_transform(X, y_pred)

            dy_dx_pred = torch.autograd.grad(y_pred, X, grad_outputs=torch.ones_like(y_pred),
                                             retain_graph=False, create_graph=False, only_inputs=True)[0]
            return [y_pred, dy_dx_pred]


# util
def get_default_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


if __name__ == "__main__":
    # Test GPU
    device = get_default_device()
    print(device)
