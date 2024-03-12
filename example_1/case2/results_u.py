import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import time


import utils


####################### Load test data #######################
data = sio.loadmat("./data/data.mat")
t_test = data["t_test"]


####################### Define basis functions #######################
kl = sio.loadmat("./data/kl_expansion.mat")
eigs = kl["eigs"].flatten()
ws = kl["ws"].flatten()
make_basis = lambda inputs: utils.make_basis(inputs, ws=ws, eigs=eigs, b=10)
make_basis_x = lambda inputs: utils.make_basis_x(inputs, ws=ws, eigs=eigs, b=10)
make_basis_xx = lambda inputs: utils.make_basis_xx(inputs, ws=ws, eigs=eigs, b=10)
make_basis_xxxx = lambda inputs: utils.make_basis_xxxx(inputs, ws=ws, eigs=eigs, b=10)


####################### Load results of HJ #######################
hj = sio.loadmat("./outputs/HJ.mat")
Phi_u = make_basis(t_test)

## case 1
q1s = hj["q1s"]
gamma1s = hj["gamma1s"]
u50s = []
u100s = []
u150s = []
for i in range(q1s.shape[0]):
    u_mu = Phi_u @ q1s[i, ...]
    u50s += [u_mu[50]]
    u100s += [u_mu[100]]
    u150s += [u_mu[150]]

## case 2
q2s = hj["q2s"]
gamma2s = hj["gamma2s"]
for i in range(q2s.shape[0]):
    u_mu = Phi_u @ q2s[i, ...]
    u50s += [u_mu[50]]
    u100s += [u_mu[100]]
    u150s += [u_mu[150]]

## case 3
q3s = hj["q3s"]
gamma3s = hj["gamma3s"]
for i in range(q3s.shape[0]):
    u_mu = Phi_u @ q3s[i, ...]
    u50s += [u_mu[50]]
    u100s += [u_mu[100]]
    u150s += [u_mu[150]]

## case 4
q4s = hj["q4s"]
gamma4s = hj["gamma4s"]
for i in range(q4s.shape[0]):
    u_mu = Phi_u @ q4s[i, ...]
    u50s += [u_mu[50]]
    u100s += [u_mu[100]]
    u150s += [u_mu[150]]

## case 5
q5s = hj["q5s"]
gamma5s = hj["gamma5s"]
for i in range(q5s.shape[0]):
    u_mu = Phi_u @ q5s[i, ...]
    u50s += [u_mu[50]]
    u100s += [u_mu[100]]
    u150s += [u_mu[150]]


####################### Save results of SciML #######################
gammas = np.concatenate(
    [gamma1s, gamma2s, gamma3s, gamma4s, gamma5s], axis=-1
).flatten()
u50s = np.array(u50s).flatten()
u100s = np.array(u100s).flatten()
u150s = np.array(u150s).flatten()
idx = np.argsort(gammas)

plt.plot(np.log(gammas[idx]), u50s[idx])
plt.show()
plt.plot(np.log(gammas[idx]), u100s[idx])
plt.show()
plt.plot(np.log(gammas[idx]), u150s[idx])
plt.show()


sio.savemat(
    "./outputs/results_u.mat",
    {
        "gammas": gammas,
        "u50s": u50s,
        "u100s": u100s,
        "u150s": u150s,
        "sigmas": np.sqrt(1 / gammas),
    },
)
