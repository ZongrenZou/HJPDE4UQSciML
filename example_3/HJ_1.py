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

idx1 = [0,1,2,3,4,5,6]
idx2 = [e + 7 for e in idx1]
idx2.reverse()
idx = idx1 + idx2
print(idx1, flush=True)
print(idx2, flush=True)
for i in range(2):
    idx_odd = [e + (i+1) * 14 for e in idx1]
    idx_even = [e + (i+1) * 14 for e in idx2]
    print(idx_odd, flush=True)
    print(idx_even, flush=True)
    idx += idx_odd + idx_even
i = i + 1
idx_odd = [e + (i+1) * 14 for e in idx1]
print(idx_odd, flush=True)
idx += idx_odd


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
    "./outputs/HJ_1.mat",
    {
        "Ps": Ps,
        "qs": qs, 
    }
)
