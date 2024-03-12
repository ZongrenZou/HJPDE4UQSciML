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


#################### Load data #########################
data = sio.loadmat("./data/train.mat")
x_train = data["x_train"]
f_train = data["f_train"]
data = sio.loadmat("./data/test.mat")
x_test = data["x_test"]
u_test = data["u_test"]
f_test = data["f_test"]


####################### Define basis functions #######################
n = 50
noise_f = 2
make_basis = lambda inputs: utils.make_basis(inputs, n=n)
make_basis_f = lambda inputs: utils.make_basis_f(inputs, n=n)


####################### Start streaming data of f #######################
update = jax.jit(utils.update)
P0 = jnp.eye(n)
q0 = jnp.zeros([n, 1])


P = P0.copy()
q = q0.copy()
h = 0.1
Ps = []
qs = []

t0 = time.time()
for i in range(x_train.shape[0]):
    # compute Phi and b for the new data
    Phi = jnp.array(make_basis_f(x_train[i : i + 1]))
    b = jnp.array(f_train[i : i + 1])

    for j in range(int(1 / h)):
        P, q = update(P, q, h, Phi, b, lamb=1 / noise_f**2)

    if (i + 1) % 1000 == 0:
        # save every 1000 iterations
        t1 = time.time()
        print("Adapting {}th data point".format(str(i + 1)))
        print("Elapsed: ", t1 - t0)
        t0 = time.time()
        Ps += [P]
        qs += [q]


####################### Save results #######################
Ps = jnp.stack(Ps, axis=0)
qs = jnp.stack(qs, axis=0)

sio.savemat(
    "./outputs/HJ.mat",
    {
        "Ps": Ps,
        "qs": qs,
    },
)


####################### Make predictions #######################
Phi_u = make_basis(x_test)
Phi_f = make_basis_f(x_test)

u_mu_1 = Phi_u @ qs[0]
f_mu_1 = Phi_f @ qs[0]

u_sd_1 = np.sqrt(np.diagonal(Phi_u @ Ps[0] @ Phi_u.T)).reshape([-1, 1])
f_sd_1 = np.sqrt(np.diagonal(Phi_f @ Ps[0] @ Phi_f.T)).reshape([-1, 1])

u_mu_2 = Phi_u @ qs[4]
f_mu_2 = Phi_f @ qs[4]

u_sd_2 = np.sqrt(np.diagonal(Phi_u @ Ps[4] @ Phi_u.T)).reshape([-1, 1])
f_sd_2 = np.sqrt(np.diagonal(Phi_f @ Ps[4] @ Phi_f.T)).reshape([-1, 1])

u_mu_3 = Phi_u @ qs[-1]
f_mu_3 = Phi_f @ qs[-1]

u_sd_3 = np.sqrt(np.diagonal(Phi_u @ Ps[-1] @ Phi_u.T)).reshape([-1, 1])
f_sd_3 = np.sqrt(np.diagonal(Phi_f @ Ps[-1] @ Phi_f.T)).reshape([-1, 1])


sio.savemat(
    "./outputs/results.mat",
    {
        "x_test": x_test,
        "u_test": u_test,
        "f_test": f_test,
        "x_train": x_train,
        "f_train": f_train,
        "u_mu_1": u_mu_1,
        "u_sd_1": u_sd_1,
        "f_mu_1": f_mu_1,
        "f_sd_1": f_sd_1,
        "u_mu_2": u_mu_2,
        "u_sd_2": u_sd_2,
        "f_mu_2": f_mu_2,
        "f_sd_2": f_sd_2,
        "u_mu_3": u_mu_3,
        "u_sd_3": u_sd_3,
        "f_mu_3": f_mu_3,
        "f_sd_3": f_sd_3,
    },
)
