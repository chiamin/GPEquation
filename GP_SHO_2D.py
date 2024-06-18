import sys, copy
sys.path.append('/home/chiamin/project/2023/qtt/code/new/tools/')
import numpy_dmrg as dmrg
import numpy as np
import matplotlib.pyplot as plt
import polynomial as poly
import differential as diff
import npmps
import plot_utility as ptut
import hamilt.hamilt_sho as sho
import gradient_descent as gd
import qtt_tools as qtt

def psi_sqr (psi):
    psi_op = npmps.MPS_to_MPO (psi)
    res = npmps.exact_apply_MPO (psi_op, psi)
    res = npmps.svd_compress_MPS (res, cutoff=1e-12)
    return res

# psi2 is the initial guess of the result
def fit_psi_sqr (psi, psi2):
    psi_op = npmps.MPS_to_MPO (psi)
    fit, overlap = dmrg.fit_apply_MPO (psi_op, psi, psi2, numCenter=1, nsweep=1, cutoff=1e-12)
    return fit

def make_H_GP (H0, psi, psi2, g):
    psi2 = psi_sqr (psi)
    #psi2 = fit_psi_sqr (psi, psi2)

    H_psi = npmps.MPS_to_MPO (psi2)
    H_psi[0] *= g
    H = npmps.sum_2MPO (H0, H_psi)
    H = npmps.svd_compress_MPO (H, cutoff=1e-12)
    return H, psi2

def imag_time_evol (H0, psi, g, dt, steps, maxdim, cutoff=1e-12):
    psi = copy.copy(psi)
    psi2 = psi_sqr (psi)
    enss = []
    for n in range(steps):
        # Update the Hamiltonian
        H, psi2 = make_H_GP (H0, psi, psi2, g)
        # TDVP
        psi, ens, terrs = dmrg.tdvp (1, psi, H, dt, [maxdim], cutoff=cutoff, krylovDim=20, verbose=False)
        en = ens[-1]*dx
        enss.append(en)
        print('TDVP',n,en)

    return psi, enss

def gradient_descent (H0, psi, g, gamma, steps):
    psi = copy.copy(psi)
    psi2 = psi_sqr (psi)
    enss = []
    for n in range(steps):
        # Update the Hamiltonian
        H, psi2 = make_H_GP (H0, psi, psi2, g)
        # Gradient descent
        psi, en = gd.gradient_descent (psi, H, gamma)
        en *= dx
        enss.append(en)
        print('GD',n,en)
    return psi, enss

if __name__ == '__main__':
    N = 5
    x1,x2 = -10,10

    Ndx = 2**N-1
    dx = (x2-x1)/Ndx
    print('dx',dx)

    g = 62.742 / dx

    maxdim = 20
    cutoff = 1e-12

    H0 = sho.make_H (N, x1, x2)
    H0 = npmps.get_H_2D (H0)

    # Initial MPS
    psi = npmps.random_MPS (2*N,2,maxdim)
    psi = npmps.to_canonical_form (psi, 0)
    psi[0] /= np.linalg.norm(psi[0])

    psi, ens, terrs = dmrg.dmrg (1, psi, H0, [10]*10, cutoff=1e-12, krylovDim=10, verbose=True)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ptut.plot_2D (psi, x1, x2, ax=ax, title='Init')

    # TDVP
    dt = dx**2*0.5
    print('dt',dt)
    psi_TDVP, ens_TDVP = imag_time_evol (H0, psi, g, dt, steps=20, maxdim=maxdim, cutoff=cutoff)

    # Gradient descent
    gamma = dt*0.05
    print('gamma',gamma)
    psi_GD, ens_GD = gradient_descent (H0, psi, g, gamma, steps=100)

    '''# Grow site
    for i in range(1,2):
        dx *= 0.5
        g *= 2
        gamma *= 0.1
        H02 = sho.make_H (N+i, x1, x2)
        print(len(H02))
        H02 = npmps.get_H_2D (H02)
        psi_GD2 = qtt.grow_site_2D (psi_GD)


        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ptut.plot_2D (psi_GD2, x1, x2, ax=ax, title='Init')
        plt.show()

        print(len(H02),len(psi_GD2))
        psi_GD2, ens_GD2 = gradient_descent (H02, psi_GD2, g, gamma, steps=100)'''


    # Plot energy
    fig2, ax2 = plt.subplots()
    ax2.plot(range(len(ens_TDVP)), ens_TDVP, label='TDVP')
    ax2.plot(range(len(ens_GD)), ens_GD, label='GD')
    #ax2.plot(range(len(ens_GD2)), ens_GD, label='GD2')
    ax2.legend()


    # Plot wavefunction
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    X, Y, Z = ptut.plot_2D (psi, x1, x2, ax=ax, label='Init')
    # 
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    X_TDVP, Y_TDVP, Z_TDVP = ptut.plot_2D (psi_TDVP, x1, x2, ax=ax, label='TDVP')
    ax.legend()
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    X_GD, Y_GD, Z_GD = ptut.plot_2D (psi_GD, x1, x2, ax=ax, label='GD')
    ax.legend()

    fig, ax = plt.subplots()
    y = 2**N//2
    ax.plot (X[y,:], Z[y,:], label='Init')
    ax.plot (X_TDVP[y,:], Z_TDVP[y,:], label='TDVP')
    ax.plot (X_GD[y,:], Z_GD[y,:], label='GD')
    ax.legend()


    plt.show()
