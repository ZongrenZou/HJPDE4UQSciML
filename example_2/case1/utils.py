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
def make_basis(x, n):
    M = np.linspace(1, n, n).reshape([1, -1])
    Z = x @ M
    basis = np.sqrt(2) * np.sin(np.pi * Z) / M / np.pi
    return basis


def make_basis_f(x, n):
    D = 0.001
    kappa = 0.1
    M = np.linspace(1, n, n).reshape([1, -1])
    Z = x @ M
    basis_x = np.sqrt(2) * np.cos(np.pi * Z)
    basis_xx = -np.sqrt(2) * np.sin(np.pi * Z) * M * np.pi
    return D * basis_xx + kappa * basis_x
