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


#################### Load data #########################
data = sio.loadmat("./data/train.mat")
xx = data["x_test"]
yy = data["y_test"]
f_test = data["f_test"]
u_test = data["u_test"]
f_train = data["f_train"]
noise_f = 0.5


#################### Determine the order #########################
#### Divide the domain into multiple blocks
x_blocks = []
y_blocks = []
f_blocks = []
s = 64
for i in range(7):
    for j in range(7):
        x_blocks += [xx[1:-1, 1:-1][i*s:(i+1)*s, j*s:(j+1)*s].reshape([-1, 1])]
        y_blocks += [yy[1:-1, 1:-1][i*s:(i+1)*s, j*s:(j+1)*s].reshape([-1, 1])]
        f_blocks += [f_train[1:-1, 1:-1][i*s:(i+1)*s, j*s:(j+1)*s].reshape([-1, 1])]

idx = [
    3*7 + 3,
    1*7 + 1, 1*7 + 3, 1*7 + 5, 3*7 + 5, 5*7 + 5, 5*7 + 3, 5*7 + 1, 3*7 + 1,
    0*7 + 0, 0*7 + 2, 2*7 + 2, 2*7 + 0,
    0*7 + 4, 0*7 + 6, 2*7 + 6, 2*7 + 4,
    4*7 + 4, 4*7 + 6, 6*7 + 6, 6*7 + 4,
    4*7 + 0, 4*7 + 2, 6*7 + 2, 6*7 + 0,
    0*7 + 1, 1*7 + 2, 2*7 + 1, 1*7 + 0,
    0*7 + 3, 1*7 + 4, 2*7 + 3,
    0*7 + 5, 1*7 + 6, 2*7 + 5,
    3*7 + 6, 4*7 + 5, 3*7 + 4,
    5*7 + 6, 6*7 + 5, 5*7 + 4,
    4*7 + 3, 6*7 + 3, 5*7 + 2,
    4*7 + 1, 6*7 + 1, 5*7 + 0,
    3*7 + 2, 3*7 + 0,
]


####################### Define basis functions #######################
n = 225
make_basis = lambda _x, _y: utils.make_basis(_x, _y, n=np.sqrt(n).astype(np.int32))
make_basis_f = lambda _x, _y: utils.make_basis_f(_x, _y, n=np.sqrt(n).astype(np.int32))


####################### Start #######################
P0 = jnp.eye(n)
q0 = jnp.zeros([n, 1])
Ps = [P0]
qs = [q0]
update = jax.jit(utils.update)
h = 2e-6


_t0 = time.time()
for k in idx:
    Phi = make_basis_f(x_blocks[k], y_blocks[k])
    Phi = jnp.array(Phi.reshape([-1, n]))
    b = jnp.array(f_blocks[k])

    P, q = Ps[-1], qs[-1]
    t0 = time.time()
    for j in range(int(1/h)):
        P, q = update(P, q, h, Phi, b, lamb=1/noise_f**2)
    t1 = time.time()
    print(k, t1 - t0, q.flatten()[0], flush=True)

    Ps += [P]
    qs += [q]

Ps = jnp.stack(Ps[1:], axis=0)
qs = jnp.stack(qs[1:], axis=0)
print(Ps.shape, qs.shape, flush=True)
print("Elapsed: ", time.time() - _t0, flush=True)


####################### Save results of HJ #######################
sio.savemat(
    "./outputs/HJ_2.mat",
    {
        "Ps": Ps,
        "qs": qs, 
    }
)
