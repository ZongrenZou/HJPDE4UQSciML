import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from jax import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import time


import utils


####################### Load data #######################
data = sio.loadmat("./data/data.mat")
t_f_train = data["t_train"]
f_train = data["f_train"]
t_new_train = data["t_new_train"]
f_new_train = data["f_new_train"]
noise_f = data["noise_f"]
noise_u = data["noise_u"]
noise_u_t = data["noise_u_t"]

t0 = np.array([0]).reshape([-1, 1])
t1 = np.array([1]).reshape([-1, 1])
u0 = data["u0"]
u1 = data["u1"]
u0_t = data["u0_t"]
u1_t = data["u1_t"]

t_u_train = np.concatenate([t0, t1], axis=0)
u_train = np.concatenate([u0, u1], axis=0)
u_t_train = np.concatenate([u0_t, u1_t], axis=0)

data = sio.loadmat("./data/data_test.mat")
t_test = data["t_test"]
f_test = data["f_test"]
u_test = data["u_test"]


####################### Define basis functions #######################
kl = sio.loadmat("./data/kl_expansion.mat")
eigs = kl["eigs"].flatten()
ws = kl["ws"].flatten()
make_basis = lambda inputs: utils.make_basis(inputs, ws=ws, eigs=eigs, b=10)
make_basis_x = lambda inputs: utils.make_basis_x(inputs, ws=ws, eigs=eigs, b=10)
make_basis_xx = lambda inputs: utils.make_basis_xx(inputs, ws=ws, eigs=eigs, b=10)
make_basis_xxxx = lambda inputs: utils.make_basis_xxxx(inputs, ws=ws, eigs=eigs, b=10)


####################### Initial training #######################
"""
Perform the initial training with the method of least squares.
Note that this could be done with the Riccati-based approach as well.
"""
sigma = 1
# sigma denotes the standard deviation of the prior
Gamma = np.eye(ws.shape[0]) / sigma**2
Phi_u = make_basis(t_u_train)
Phi_u_t = make_basis_x(t_u_train)
Phi_f = (
    0.0001 * make_basis_xxxx(t_f_train)
    + 0.01 * make_basis_xx(t_f_train)
    + make_basis(t_f_train)
)

A = (
    Gamma
    + 1 / noise_u / noise_u * Phi_u.T @ Phi_u
    + 1 / noise_u_t / noise_u_t * Phi_u_t.T @ Phi_u_t
    + 1 / noise_f / noise_f * Phi_f.T @ Phi_f
)
b = (
    1 / noise_u / noise_u * Phi_u.T @ u_train
    + 1 / noise_u_t / noise_u_t * Phi_u_t.T @ u_t_train
    + 1 / noise_f / noise_f * Phi_f.T @ f_train
)

P0 = np.linalg.inv(A)
q0 = np.linalg.solve(A, b)


####################### Remove one data point #######################
Phi = (
    0.0001 * make_basis_xxxx(t_new_train[0:1])
    + 0.01 * make_basis_xx(t_new_train[0:1])
    + make_basis(t_new_train[0:1])
)
b = f_new_train[0:1]

Phi = jnp.array(Phi)
b = jnp.array(b)

P1 = jnp.array(P0.copy())
q1 = jnp.array(q0.copy())
update = jax.jit(utils.update)
h = 1e-5

for j in range(int(1 / h)):
    # solving the Riccati ODEs backward
    P1, q1 = update(P1, q1, -h, Phi, b, lamb=1 / noise_f**2)


####################### Remove the other data point #######################
Phi = (
    0.0001 * make_basis_xxxx(t_new_train[1:2])
    + 0.01 * make_basis_xx(t_new_train[1:2])
    + make_basis(t_new_train[1:2])
)
b = f_new_train[1:2]

Phi = jnp.array(Phi)
b = jnp.array(b)

P2 = jnp.array(P1.copy())
q2 = jnp.array(q1.copy())
h = 1e-5

for j in range(int(1 / h)):
    # solving the Riccati ODEs backward
    P2, q2 = update(P2, q2, -h, Phi, b, lamb=1 / noise_f**2)


####################### Save results of HJ #######################
sio.savemat(
    "./outputs/HJ.mat",
    {
        "P0": P0,
        "q0": q0,
        "P1": P1,
        "q1": q1,
        "P2": P2,
        "q2": q2,
    },
)
