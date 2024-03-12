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


####################### Load results of the initial training #######################
results_0 = sio.loadmat("./outputs/HJ_0.mat")
P0 = results_0["P0"]
q0 = results_0["q0"]


####################### Start tuning sigma continuously #######################
update = jax.jit(utils.update)
Phi = jnp.eye(30)
b = jnp.zeros([30, 1])

## increase sigma from 1 to 2, which is equivalent to decreasing gamma from 1 to 0.25
print("Tuning sigma from 1 to 2...")
h = 1e-5
dt = 1 - 0.25
P1, q1 = jnp.array(P0), jnp.array(q0)

P1s, q1s = [P1], [q1]
gamma1s = [1]
for i in range(int(dt / h)):
    P1, q1 = update(P1, q1, -h, Phi, b, lamb=1)  # solve the Riccati ODE backward
    if (i + 1) % 100 == 0:
        # save P, q for every 100 iterations
        P1s += [P1]
        q1s += [q1]
        gamma1s += [gamma1s[-1] - 100 * h]

## increase sigma from 2 to 5, which is equivalent to decreasing gamma from 0.25 to 0.04
print("Tuning sigma from 2 to 5...")
h = 1e-6
dt = 0.25 - 0.04
P2, q2 = P1.copy(), q1.copy()

P2s, q2s = [P2], [q2]
gamma2s = [0.25]
for i in range(int(dt / h)):
    P2, q2 = update(P2, q2, -h, Phi, b, lamb=1)  # solve the Riccati ODE backward
    if (i + 1) % 100 == 0:
        # save P, q for every 100 iterations
        P2s += [P2]
        q2s += [q2]
        gamma2s += [gamma2s[-1] - 100 * h]

## increase sigma from 5 to 10, which is equivalent to decreasing gamma from 0.04 to 0.01
print("Tuning sigma from 5 to 10...")
h = 1e-7
dt = 0.04 - 0.01
P3, q3 = P2.copy(), q2.copy()

P3s, q3s = [P3], [q3]
gamma3s = [0.04]
for i in range(int(dt / h)):
    P3, q3 = update(P3, q3, -h, Phi, b, lamb=1)  # solve the Riccati ODE backward
    if (i + 1) % 100 == 0:
        P3s += [P3]
        q3s += [q3]
        gamma3s += [gamma3s[-1] - 100 * h]

## increase sigma from 10 to 20, which is equivalent to decreasing gamma from 0.01 to 0.0025
print("Tuning sigma from 10 to 20...")
h = 1e-7
dt = 0.01 - 0.0025
P4, q4 = P3.copy(), q3.copy()

P4s, q4s = [P4], [q4]
gamma4s = [0.01]
for i in range(int(dt / h)):
    P4, q4 = update(P4, q4, -h, Phi, b, lamb=1)  # solve the Riccati ODE backward
    if (i + 1) % 100 == 0:
        P4s += [P4]
        q4s += [q4]
        gamma4s += [gamma4s[-1] - 100 * h]

## decrease sigma from 1 to 0.5, which is equivalent to increasing gamma from 1 to 4
print("Tuning sigma from 1 to 0.5...")
h = 1e-5
dt = 4 - 1
P5, q5 = P0.copy(), q0.copy()

P5s, q5s = [P5], [q5]
gamma5s = [1]
for i in range(int(dt / h)):
    P5, q5 = update(P5, q5, h, Phi, b, lamb=1)  # solve the Riccati ODE forward
    if (i + 1) % 100 == 0:
        P5s += [P5]
        q5s += [q5]
        gamma5s += [gamma5s[-1] + 100 * h]


####################### Save results of HJ #######################
P1s, q1s, gamma1s = np.array(P1s), np.array(q1s), np.array(gamma1s)
P2s, q2s, gamma2s = np.array(P2s), np.array(q2s), np.array(gamma2s)
P3s, q3s, gamma3s = np.array(P3s), np.array(q3s), np.array(gamma3s)
P4s, q4s, gamma4s = np.array(P4s), np.array(q4s), np.array(gamma4s)
P5s, q5s, gamma5s = np.array(P5s), np.array(q5s), np.array(gamma5s)

print(q1s.shape, q2s.shape, q3s.shape, q4s.shape, q5s.shape)
sio.savemat(
    "./outputs/HJ.mat",
    {
        "P1s": P1s,
        "q1s": q1s,
        "gamma1s": gamma1s,
        "P2s": P2s,
        "q2s": q2s,
        "gamma2s": gamma2s,
        "P3s": P3s,
        "q3s": q3s,
        "gamma3s": gamma3s,
        "P4s": P4s,
        "q4s": q4s,
        "gamma4s": gamma4s,
        "P5s": P5s,
        "q5s": q5s,
        "gamma5s": gamma5s,
    },
)
