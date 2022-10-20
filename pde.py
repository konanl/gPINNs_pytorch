#!/usr/bin/env python
# encoding: utf-8
"""
    @author: LiangL. Yan
    @license: (C) Copyright forever
    @contact: yokingyan@gmail.com
    @ide: pycharm
    @file: pde.py
    @time: 2022/10/4 11:15
    @desc:
    
"""
import numpy as np


########## Function ##########
import torch


def function_sol(x):
    return -(1.4 - 3 * x) * np.sin(18 * x)


def function_sol_grad(x):
    return 3 * np.sin(18 * x) + 18 * (3 * x - 1.4) * np.cos(18 * x)


########## Poisson 1D ##########
def poisson_sol(x):
    """Poisson 1D analytical solution"""
    solution = x + 1 / 8 * np.sin(8 * x)
    for i in range(1, 5):
        solution += 1 / i * np.sin(i * x)

    return solution


def poisson_sol_grad(x):
    """Gradient of poisson 1D analytical solution"""
    solution = 1 + np.cos(8 * x)
    for i in range(1, 5):
        solution += np.cos(i * x)
    return solution


########## Diffusion Reaction Equation ##########
def diffusion_reaction_sol(a):
    """The analytic solution of diffusion reaction equation"""
    x, t = a[:, 0:1], a[:, 1:2]
    val = np.sin(8 * x) / 8
    for i in range(1, 5):
        val += np.sin(i * x) / i
    return np.exp(-t) *val


def icfunc(x):
    """The ic u(x, 0)"""
    return (
        torch.sin(8 * x) / 8
        + torch.sin(1 * x) / 1
        + torch.sin(2 * x) / 2
        + torch.sin(3 * x) / 3
        + torch.sin(4 * x) / 4
    )


def dudt(x):
    x_in, t_in = x[:, 0:1], x[:, 1:2]

    val = np.sin(8 * x_in) / 8
    for i in range(1, 5):
        val += np.sin(i * x_in) / i
    return -np.exp(-t_in) * val


def dudx(x):
    x_in, t_in = x[:, 0:1], x[:, 1:2]

    val = np.cos(8 * x_in)
    for i in range(1, 5):
        val += np.cos(i * x_in)
    return np.exp(-t_in) * val

########## Brinkman-Forchheimer ##########
g = 1
v = 1e-3
K = 1e-3
e = 0.4
H = 1


def BF_sol(x):
    r = (v * e / (1e-3 * K)) ** (0.5)
    return g * K / v * (1 - np.cosh(r * (x - H / 2)) / np.cosh(r * H / 2))


def BF_grad(x):
    r = (v * e / (1e-3 * K)) ** (0.5)
    return g * K / v * r * (-np.sinh(r * (x - H / 2)) / np.cosh(r * H / 2))

