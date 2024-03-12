import numpy as np


####################### Define RK4 update #######################
def update(P0, q0, h, phi, b, lamb=1):
    """One RK4 update. JAX XLA (jax.jit) may be applied."""
    Pk1 = -lamb * (P0).T @ phi.T @ phi @ P0
    qk1 = -lamb * (P0).T @ phi.T @ (phi @ q0 - b)
    Pk2 = -lamb * (P0 + 0.5 * h * Pk1).T @ phi.T @ phi @ (P0 + 0.5 * h * Pk1)
    qk2 = -lamb * (P0 + 0.5 * h * Pk1).T @ phi.T @ (phi @ (q0 + 0.5 * h * qk1) - b)
    Pk3 = -lamb * (P0 + 0.5 * h * Pk2).T @ phi.T @ phi @ (P0 + 0.5 * h * Pk2)
    qk3 = -lamb * (P0 + 0.5 * h * Pk2).T @ phi.T @ (phi @ (q0 + 0.5 * h * qk2) - b)
    Pk4 = -lamb * (P0 + h * Pk3).T @ phi.T @ phi @ (P0 + h * Pk3)
    qk4 = -lamb * (P0 + h * Pk3).T @ phi.T @ (phi @ (q0 + h * qk3) - b)
    P = P0 + h * (Pk1 + 2 * Pk2 + 2 * Pk3 + Pk4) / 6
    q = q0 + h * (qk1 + 2 * qk2 + 2 * qk3 + qk4) / 6
    return P, q


####################### Define basis functions for the KL expansion #######################
def make_basis(x, ws, eigs, b):
    n = eigs.shape[0]
    N = x.shape[0]
    Phi = np.zeros(shape=[N, n])
    for i in range(1, ws.shape[0] + 1):
        if i % 2 == 0:
            Phi[:, i - 1 : i] = np.sin(ws[i - 1] * x) / np.sqrt(
                b - np.sin(2 * ws[i - 1] * b) / 2 / ws[i - 1]
            )
        else:
            Phi[:, i - 1 : i] = np.cos(ws[i - 1] * x) / np.sqrt(
                b + np.sin(2 * ws[i - 1] * b) / 2 / ws[i - 1]
            )
        Phi[:, i - 1] = np.sqrt(eigs[i - 1]) * Phi[:, i - 1]
    return Phi


def make_basis_x(x, ws, eigs, b):
    n = eigs.shape[0]
    N = x.shape[0]
    Phi_x = np.zeros(shape=[N, n])
    for i in range(1, ws.shape[0] + 1):
        if i % 2 == 0:
            Phi_x[:, i - 1 : i] = (
                ws[i - 1]
                * np.cos(ws[i - 1] * x)
                / np.sqrt(b - np.sin(2 * ws[i - 1] * b) / 2 / ws[i - 1])
            )
        else:
            Phi_x[:, i - 1 : i] = (
                -ws[i - 1]
                * np.sin(ws[i - 1] * x)
                / np.sqrt(b + np.sin(2 * ws[i - 1] * b) / 2 / ws[i - 1])
            )
        Phi_x[:, i - 1] = np.sqrt(eigs[i - 1]) * Phi_x[:, i - 1]
    return Phi_x


def make_basis_xx(x, ws, eigs, b):
    n = eigs.shape[0]
    N = x.shape[0]
    Phi_xx = np.zeros(shape=[N, n])
    for i in range(1, ws.shape[0] + 1):
        if i % 2 == 0:
            Phi_xx[:, i - 1 : i] = (
                -ws[i - 1] ** 2
                * np.sin(ws[i - 1] * x)
                / np.sqrt(b - np.sin(2 * ws[i - 1] * b) / 2 / ws[i - 1])
            )
        else:
            Phi_xx[:, i - 1 : i] = (
                -ws[i - 1] ** 2
                * np.cos(ws[i - 1] * x)
                / np.sqrt(b + np.sin(2 * ws[i - 1] * b) / 2 / ws[i - 1])
            )
        Phi_xx[:, i - 1] = np.sqrt(eigs[i - 1]) * Phi_xx[:, i - 1]
    return Phi_xx


def make_basis_xxxx(x, ws, eigs, b):
    n = eigs.shape[0]
    N = x.shape[0]
    Phi_xxxx = np.zeros(shape=[N, n])
    for i in range(1, ws.shape[0] + 1):
        if i % 2 == 0:
            Phi_xxxx[:, i - 1 : i] = (
                ws[i - 1] ** 4
                * np.sin(ws[i - 1] * x)
                / np.sqrt(b - np.sin(2 * ws[i - 1] * b) / 2 / ws[i - 1])
            )
        else:
            Phi_xxxx[:, i - 1 : i] = (
                ws[i - 1] ** 4
                * np.cos(ws[i - 1] * x)
                / np.sqrt(b + np.sin(2 * ws[i - 1] * b) / 2 / ws[i - 1])
            )
        Phi_xxxx[:, i - 1] = np.sqrt(eigs[i - 1]) * Phi_xxxx[:, i - 1]
    return Phi_xxxx
