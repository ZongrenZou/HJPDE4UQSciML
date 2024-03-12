import numpy as np
import scipy.io as sio


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


####################### Load results of HJ #######################
hj_results = sio.loadmat("./HJ.mat")
Ps = hj_results["Ps"]
qs = hj_results["qs"]

print(Ps.shape, qs.shape)

A_u_test = make_basis(t_test)
A_f_test = (
    0.0001 * make_basis_xxxx(t_test) + 0.01 * make_basis_xx(t_test) + make_basis(t_test)
)


####################### Produce results for SciML #######################
u_mus = []
f_mus = []
u_sds = []
f_sds = []
for i in range(qs.shape[0]):
    u_mus += [A_u_test @ qs[i, ...]]
    f_mus += [A_f_test @ qs[i, ...]]
    u_sds += [np.sqrt(np.diagonal(A_u_test @ Ps[i, ...] @ A_u_test.T))]
    f_sds += [np.sqrt(np.diagonal(A_f_test @ Ps[i, ...] @ A_f_test.T))]


sio.savemat(
    "./outputs/results.mat",
    {
        "t_test": t_test,
        "u_test": u_test,
        "f_test": f_test,
        "t_u_train": t_u_train,
        "u_train": u_train,
        "u_t_train": u_t_train,
        "t_f_train": t_f_train,
        "f_train": f_train,
        "u_mus": u_mus,
        "f_mus": f_mus,
        "u_sds": u_sds,
        "f_sds": f_sds,
    },
)
