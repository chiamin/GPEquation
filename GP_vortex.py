import sys, copy
sys.path.append('/home/chiamin/NumpyTensorTools/')
import numpy_dmrg as dmrg
import numpy as np
import matplotlib.pyplot as plt
import polynomial as poly
import differential as diff
import npmps
import plot_utility as pltut
import hamilt.hamilt_sho as sho
import gradient_descent as gd
import qtt_tools as qtt
import hamilt.hamilt_angular_momentum as ang
import tci
import time

def psi_sqr (psi):
    psi_op = qtt.MPS_to_MPO (psi)
    psi_op = npmps.conj (psi_op)
    res = npmps.exact_apply_MPO (psi_op, psi)

    #print('psi2 dim, before compression',npmps.MPS_dims(res))
    res = npmps.svd_compress_MPS (res, cutoff=1e-12)
    #print('psi2 dim, after compression',npmps.MPS_dims(res))

    return res

# psi2 is the initial guess of the result
def fit_psi_sqr (psi, psi2, maxdim, cutoff=1e-16):
    psi_op = qtt.MPS_to_MPO (psi)
    psi_op = npmps.conj (psi_op)
    fit = dmrg.fit_apply_MPO (psi_op, psi, psi2, numCenter=1, nsweep=1, maxdim=maxdim, cutoff=cutoff)



    #psi2_exact = psi_sqr (psi)
    #print_overlap (psi2_exact, fit)

    return fit

def make_H_GP (H0, psi, psi2, g, maxdim):
    t1 = time.time()                                # time
    #psi2 = psi_sqr (psi)
    psi2 = fit_psi_sqr (psi, psi2, 4*maxdim)
    t2 = time.time()                                # time
    print('psi2 time',(t2-t1))                      # time

    H_psi = qtt.MPS_to_MPO (psi2)
    H_psi[0] *= 2*g
    H = npmps.sum_2MPO (H0, H_psi)

    # Rescale H
    for i in range(len(H)):
        H[i] *= 8e-1

    return H, psi2

def imag_time_evol (H0, psi, g, dt, steps, maxdim, cutoff=1e-12, krylovDim=10):
    psi = copy.copy(psi)
    psi2 = psi_sqr (psi)
    enss = []
    for n in range(steps):
        t1 = time.time()                                # time
        # Update the Hamiltonian
        H, psi2 = make_H_GP (H0, psi, psi2, g, maxdim=maxdim)

        # TDVP
        psi, ens, terrs = dmrg.tdvp (1, psi, H, dt, [maxdim], cutoff=cutoff, krylovDim=krylovDim, verbose=False)
        en = ens[-1]*dx
        enss.append(en)
        print('TDVP',n,en)
        t2 = time.time()                                # time
        print('imag time evol time',(t2-t1))                      # time

    return psi, enss

def gradient_descent (H0, psi, g, gamma, steps, maxdim, cutoff):
    psi = copy.copy(psi)
    psi2 = psi_sqr (psi)
    enss = []
    for n in range(steps):
        t1 = time.time()                                # time
        # Update the Hamiltonian
        H, psi2 = make_H_GP (H0, psi, psi2, g, maxdim)

        # Gradient descent
        psi, en = gd.gradient_descent (psi, H, gamma, maxdim=maxdim, cutoff=cutoff)
        en *= dx
        enss.append(en)
        print('GD',n,en)
        t2 = time.time()                                # time
        print('GD time',(t2-t1))                      # time
    return psi, enss

def fitfun_xy (x, y):
    f = np.sqrt(1/np.pi)*(x+1j*y)*np.exp(-(x**2+y**2)/2)*np.exp(2*np.pi*1j*np.random.rand())
    return f

def inds_to_num (inds, dx, shift):
    bstr = ''
    for i in inds:
        bstr += str(i)
    return pltut.bin_to_dec (bstr, dx, shift)

def fitfun (inds):
    N = len(inds)//2
    Ndx = 2**N-1
    dx = (x2-x1)/Ndx
    shift = x1

    xinds, yinds = inds[:N], inds[N:]
    x = inds_to_num (xinds, dx, shift)
    y = inds_to_num (yinds, dx, shift)
    return fitfun_xy (x, y)

def get_init_state (N, x1, x2, maxdim):
    seed = 13
    np.random.seed(seed)

    mps = tci.tci (fitfun, 2*N, 2, maxdim=maxdim)
    mps = npmps.normalize_MPS (mps)
    return mps    

def check_hermitian (mpo):
    mm = npmps.MPO_to_matrix (mpo)
    t = np.linalg.norm(mm - mm.conj().T)
    print(t)
    assert t < 1e-10

def check_the_same (mpo1, mpo2):
    m1 = npmps.MPO_to_matrix(mpo1)
    m2 = npmps.MPO_to_matrix(mpo2)
    d = np.linalg.norm(m1-m2)
    print(d)
    assert d < 1e-10

def print_overlap (mps1, mps2):
    mps1 = copy.copy(mps1)
    mps2 = copy.copy(mps2)
    mps1 = npmps.normalize_MPS(mps1)
    mps2 = npmps.normalize_MPS(mps2)
    print('overlap',npmps.inner_MPS(mps1, mps2))

if __name__ == '__main__':    
    N = 6
    x1,x2 = -10,10

    Ndx = 2**N-1
    dx = (x2-x1)/Ndx
    print('dx',dx)

    g = 100 *2/dx**2
    omega = 0.8

    maxdim = 10
    cutoff = 1e-12
    krylovDim = 10

    H_SHO = sho.make_H (N, x1, x2)
    H_SHO = npmps.get_H_2D (H_SHO)
    H_SHO = npmps.change_dtype(H_SHO, complex)

    Lz = ang.Lz_MPO (N, x1, x2)
    Lz[0] *= -2*omega

    H0 = npmps.sum_2MPO (H_SHO, Lz)

    print('Non-interacting MPO dim, before compression:',npmps.MPO_dims(H0))
    H0 = npmps.svd_compress_MPO (H0, cutoff=1e-12)
    print('Non-interacting MPO dim:',npmps.MPO_dims(H0))


    def absSqr (a):
        return abs(a)**2
    absSqr = np.vectorize(absSqr)

    # Initial MPS
    psi = get_init_state (N, x1, x2, maxdim=20)
    print('Initial psi dim, before compression:',npmps.MPS_dims(psi))
    psi = npmps.svd_compress_MPS (psi, cutoff=1e-12)
    print('Initial psi dim:',npmps.MPS_dims(psi))
    psi = qtt.normalize_MPS_by_integral (psi, x1, x2, Dim=2)

    # TDVP
    dt = dx**2*10
    print('dt',dt)
    psi_TDVP, ens_TDVP = imag_time_evol (H0, psi, g, dt, steps=200, maxdim=maxdim, cutoff=cutoff, krylovDim=krylovDim)

    # Gradient descent
    gamma = dt*0.01
    print('gamma',gamma)
    psi_GD, ens_GD = gradient_descent (H0, psi, g, gamma, steps=600, maxdim=maxdim, cutoff=cutoff)

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
        pltut.plot_2D (psi_GD2, x1, x2, ax=ax, title='Init')
        plt.show()

        print(len(H02),len(psi_GD2))
        psi_GD2, ens_GD2 = gradient_descent (H02, psi_GD2, g, gamma, steps=100)'''


    # Plot energy
    fig2, ax2 = plt.subplots()
    ax2.plot(range(len(ens_TDVP)), ens_TDVP, label='TDVP')
    ax2.plot(range(len(ens_GD)), ens_GD, label='GD')
    #ax2.plot(range(len(ens_GD2)), ens_GD, label='GD2')
    ax2.legend()
    plt.show()

    # Plot wavefunction
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    psi = qtt.normalize_MPS_by_integral (psi, x1, x2, Dim=2)
    X, Y, Z = pltut.plot_2D (psi, x1, x2, ax=ax, func=absSqr, label='Init')
    fig.savefig('init.pdf')
    # 
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    psi_TDVP = qtt.normalize_MPS_by_integral (psi_TDVP, x1, x2, Dim=2)
    X_TDVP, Y_TDVP, Z_TDVP = pltut.plot_2D (psi_TDVP, x1, x2, ax=ax, func=absSqr, label='TDVP')
    #ax.legend()
    fig.savefig('TDVP.pdf')

    #
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    psi_GD = qtt.normalize_MPS_by_integral (psi_GD, x1, x2, Dim=2)
    X_GD, Y_GD, Z_GD = pltut.plot_2D (psi_GD, x1, x2, ax=ax, func=absSqr, label='GD')
    fig.savefig('GD.pdf')
    #ax.legend()

    fig, ax = plt.subplots()
    y = 2**N//2
    Z = absSqr(Z)
    Z_TDVP = absSqr(Z_TDVP)
    Z_GD = absSqr(Z_TDVP)
    ax.plot (X[y,:], Z[y,:], label='Init')
    ax.plot (X_TDVP[y,:], Z_TDVP[y,:], label='TDVP')
    ax.plot (X_GD[y,:], Z_GD[y,:], label='GD')
    ax.legend()

    plt.show()
