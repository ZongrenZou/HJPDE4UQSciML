import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import time


import utils


####################### Load validation data #######################
data = sio.loadmat("./data/data_val.mat")
t_f_val = data["t_train"]
f_val = data["f_train"]


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
Phi_f = (
    0.0001 * make_basis_xxxx(t_f_val)
    + 0.01 * make_basis_xx(t_f_val)
    + make_basis(t_f_val)
)


## case 1
q1s = hj["q1s"]
gamma1s = hj["gamma1s"]
f_errs1 = []
for i in range(q1s.shape[0]):
    f_mu = Phi_f @ q1s[i, ...]
    f_errs1 += [np.mean((f_mu - f_val) ** 2)]

## case 2
q2s = hj["q2s"]
gamma2s = hj["gamma2s"]
f_errs2 = []
for i in range(q2s.shape[0]):
    f_mu = Phi_f @ q2s[i, ...]
    f_errs2 += [np.mean((f_mu - f_val) ** 2)]

## case 3
q3s = hj["q3s"]
gamma3s = hj["gamma3s"]
f_errs3 = []
for i in range(q3s.shape[0]):
    f_mu = Phi_f @ q3s[i, ...]
    f_errs3 += [np.mean((f_mu - f_val) ** 2)]

## case 4
q4s = hj["q4s"]
gamma4s = hj["gamma4s"]
f_errs4 = []
for i in range(q4s.shape[0]):
    f_mu = Phi_f @ q4s[i, ...]
    f_errs4 += [np.mean((f_mu - f_val) ** 2)]

## case 5
q5s = hj["q5s"]
gamma5s = hj["gamma5s"]
f_errs5 = []
for i in range(q5s.shape[0]):
    f_mu = Phi_f @ q5s[i, ...]
    f_errs5 += [np.mean((f_mu - f_val) ** 2)]


####################### Save results of SciML #######################
gammas = np.concatenate(
    [gamma1s, gamma2s, gamma3s, gamma4s, gamma5s], axis=-1
).flatten()
f_errs = f_errs1 + f_errs2 + f_errs3 + f_errs4 + f_errs5
f_errs = np.array(f_errs).flatten()
idx = np.argsort(gammas)

plt.plot(np.log(gammas[idx]), f_errs[idx])
plt.show()

sio.savemat(
    "./outputs/results_val.mat",
    {
        "gammas": gammas,
        "f_errs": f_errs,
        "sigmas": np.sqrt(1 / gammas),
    },
)
