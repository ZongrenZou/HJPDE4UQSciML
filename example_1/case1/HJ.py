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


####################### Encode the boundary condition #######################
n = 30
Gamma = jnp.eye(n)
Phi_u = jnp.array(make_basis(t_u_train))
Phi_u_t = jnp.array(make_basis_x(t_u_train))
u_train = jnp.array(u_train)
u_t_train = jnp.array(u_t_train)
noise_u = jnp.array(noise_u)
noise_u_t = jnp.array(noise_u_t)

P0 = jnp.linalg.inv(
    Gamma
    + 1 / noise_u**2 * Phi_u.T @ Phi_u
    + 1 / noise_u_t**2 * Phi_u_t.T @ Phi_u_t,
)
q0 = P0 @ (
    1 / noise_u**2 * Phi_u.T @ u_train + 1 / noise_u_t**2 * Phi_u_t.T @ u_t_train
)


####################### Start streaming data of f #######################
Ps = [P0]
qs = [q0]
update = jax.jit(utils.update)
h = 5e-5
noise_f = jnp.array(noise_f)

for i in range(f_train.shape[0]):
    # fetch P and q from the previous state
    P, q = Ps[-1], qs[-1]
    # compute Phi and b for the new data
    Phi = (
        0.0001 * make_basis_xxxx(t_f_train[i : i + 1])
        + 0.01 * make_basis_xx(t_f_train[i : i + 1])
        + make_basis(t_f_train[i : i + 1])
    )
    b = f_train[i : i + 1]
    Phi = jnp.array(Phi)
    b = jnp.array(b)

    t0 = time.time()
    for j in range(int(1 / h)):
        P, q = update(P, q, h, Phi, b, lamb=1 / noise_f**2)
    t1 = time.time()
    print(i, t1 - t0, q.flatten()[0], flush=True)

    Ps += [P]
    qs += [q]


####################### Save results of HJ #######################
Ps = jnp.stack(Ps, axis=0)
qs = jnp.stack(qs, axis=0)
print(Ps.shape, qs.shape)

sio.savemat(
    "./outputs/HJ.mat",
    {
        "Ps": Ps,
        "qs": qs,
    },
)
