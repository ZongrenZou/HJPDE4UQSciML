import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from jax.config import config

config.update("jax_enable_x64", True)

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import jax
import jax.numpy as jnp
import time


import utils


##################### Load data #####################
data = sio.loadmat("./data/train.mat")
x_test = data["x_test"]
u_test = data["u_test"]
f_test = data["f_test"]
x_train = data["x_train"]
f_train = data["f_train"]


####################### Define basis functions #######################
n = 50
noise_f = data["noise_f"]
make_basis = lambda inputs: utils.make_basis(inputs, n=n)
make_basis_f = lambda inputs: utils.make_basis_f(inputs, n=n)


####################### Define query function #######################
def query(x, noise_f):
    # data acquisition function; considered as a black box
    u = np.sin(6*np.pi*x) * np.cos(4*np.pi*x)**2
    u_x = 6 * np.pi * np.cos(4*np.pi*x)**2 * np.cos(6*np.pi*x) - \
          8 * np.pi * np.cos(4*np.pi*x) * np.sin(4*np.pi*x) * np.sin(6*np.pi*x)
    u_xx = 32 * np.pi**2 * np.sin(4*np.pi*x)**2 * np.sin(6*np.pi*x) - \
           68 * np.pi**2 * np.cos(4*np.pi*x)**2 * np.sin(6*np.pi*x) - \
           96 * np.pi**2 * np.cos(4*np.pi*x) * np.cos(6*np.pi*x) * np.sin(4*np.pi*x)

    f = 0.001 * u_xx + 0.1 * u_x
    return f + noise_f * np.random.normal(size=f.shape)


####################### Start #######################
P0 = jnp.eye(n)
q0 = jnp.zeros([n, 1])
h = 0.01
update = jax.jit(utils.update)


x_choice = np.linspace(0, 1, 101).reshape([-1, 1])
Phi = make_basis_f(x_choice)
b = query(x_choice, noise_f)

Phi_choice = jnp.array(Phi)
b_choice = jnp.array(b)

Ps = []
qs = []
P = P0.copy()
q = q0.copy()
u_sds = []
f_sds = []
u_mus = []
f_mus = []
x_train = []
f_train = []

for i in range(x_choice.shape[0]):
    Phi_f = make_basis_f(x_choice)
    f_sd = jnp.sqrt(jnp.diagonal(Phi_f @ P @ Phi_f.T)).reshape([-1, 1])
    idx = jnp.argsort(-f_sd.flatten(), axis=-1)[0]
    print("Adding {}th point".format(str(i)))
    print("Location where the predicted uncertainty is the largest:\n", x_choice[idx].flatten())
    x_train += [x_choice[idx:idx+1, :]]
    f_train += [b_choice[idx:idx+1, :]]
    
    for j in range(int(1/h)):
        P, q = update(P, q, h, Phi_choice[idx:idx+1, :], b_choice[idx:idx+1, :], lamb=1/noise_f**2)
    Ps += [P]
    qs += [q]

    Phi_u = make_basis(x_test)
    Phi_f = make_basis_f(x_test)

    u_mu = Phi_u @ q
    f_mu = Phi_f @ q
    u_mus += [u_mu]
    f_mus += [f_mu]

    u_sd = jnp.sqrt(jnp.diagonal(Phi_u @ P @ Phi_u.T)).reshape([-1, 1])
    f_sd = jnp.sqrt(jnp.diagonal(Phi_f @ P @ Phi_f.T)).reshape([-1, 1])
    u_sds += [u_sd]
    f_sds += [f_sd]


x_train = np.array(x_train).reshape([-1, 1])
f_train = np.array(f_train).reshape([-1, 1])

u_mus = np.stack(u_mus, axis=0)
u_sds = np.stack(u_sds, axis=0)
f_mus = np.stack(f_mus, axis=0)
f_sds = np.stack(f_sds, axis=0)
Ps = np.stack(Ps, axis=0)
qs = np.stack(qs, axis=0)

sio.savemat(
    "./outputs/HJ.mat",
    {
        "Ps": Ps, "qs": qs,
    }
)

sio.savemat(
    "./outputs/results.mat",
    {
        "x_test": x_test, "u_test": u_test, "f_test": f_test,
        "x_train": x_train, "f_train": f_train,
        "u_mus": u_mus, "u_sds": u_sds,
        "f_mus": f_mus, "f_sds": f_sds,
    }
)
