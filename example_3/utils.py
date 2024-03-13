import numpy as np


def update(P0, q0, h, phi, b, lamb=1):
    """One RK4 update."""
    Pk1 = - lamb * (P0).T @ phi.T @ phi @ P0
    qk1 = - lamb * (P0).T @ phi.T @ (phi @ q0 - b)
    Pk2 = - lamb * (P0+0.5*h*Pk1).T @ phi.T @ phi @ (P0+0.5*h*Pk1)
    qk2 = - lamb * (P0+0.5*h*Pk1).T @ phi.T @ (phi @ (q0+0.5*h*qk1) - b)
    Pk3 = - lamb * (P0+0.5*h*Pk2).T @ phi.T @ phi @ (P0+0.5*h*Pk2)        
    qk3 = - lamb * (P0+0.5*h*Pk2).T @ phi.T @ (phi @ (q0+0.5*h*qk2) - b)
    Pk4 = - lamb * (P0+h*Pk3).T @ phi.T @ phi @ (P0+h*Pk3)
    qk4 = - lamb * (P0+h*Pk3).T @ phi.T @ (phi @ (q0+h*qk3) - b)
    P = P0 + h * (Pk1 + 2*Pk2 + 2*Pk3 + Pk4) / 6
    q = q0 + h * (qk1 + 2*qk2 + 2*qk3 + qk4) / 6
    return P, q


def make_basis(x, y, n=20, T=1):
    T = 2 * np.pi * T
    M = np.linspace(1, n, n).reshape([1, -1])
    X = np.matmul(x, M)
    Y = np.matmul(y, M)
    basis_x = np.sqrt(2*T) * np.sin(np.pi*X / T) / M / np.pi
    basis_y = np.sqrt(2*T) * np.sin(np.pi*Y / T) / M / np.pi
    basis = basis_x[:, None, :] * basis_y[:, :, None]
    return basis


def make_basis_f(x, y, n=20, T=1):
    T = 2 * np.pi * T
    M = np.linspace(1, n, n).reshape([1, -1])
    X = np.matmul(x, M)
    Y = np.matmul(y, M)
    basis_1 = np.sqrt(2*T) * np.sin(np.pi*X / T) / M / np.pi
    basis_2 = np.sqrt(2*T) * np.sin(np.pi*Y / T) / M / np.pi
    
    basis_1_xx = -1 * (M*np.pi/T)**2 * basis_1
    basis_2_yy = -1 * (M*np.pi/T)**2 * basis_2
    
    u = basis_1[:, None, :] * basis_2[:, :, None]
    u_xx = basis_1_xx[:, None, :] * basis_2[:, :, None]
    u_yy = basis_1[:, None, :] * basis_2_yy[:, :, None]
    
    return u - u_xx - u_yy