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

def find_ground_state (H, psi):

    # Define the bond dimensions for the sweeps
    nsweep = 2
    maxdims = [2]*nsweep + [4]*nsweep + [8]*nsweep + [16]*nsweep + [32]*nsweep
    cutoff = 1e-12

    # Run dmrg
    psi0, ens0, terrs0 = dmrg.dmrg (1, psi, H, maxdims, cutoff=1e-12, krylovDim=10, verbose=True)
    # Plot
    #mps = npmps.normalize_by_integral (psi0, x1, x2)
    #ptut.plot_1D (mps, x1, x2, ax=ax, label='DMRG')
    #for i,j in zip(ens0,terrs0): print(i,j)

    # TDVP
    dt = 0.1
    psi1, ens1, terrs1 = dmrg.tdvp (1, psi0, H, dt, maxdims, cutoff=1e-12, krylovDim=4, verbose=False)
    #for i,j in zip(ens3,terrs3): print(i,j)

    # Plot
    #mps = npmps.normalize_by_integral (psi3, x1, x2)
    #ptut.plot_1D (mps, x1, x2, ax=ax, label='TDVP')

    # Gradient descent
    #print('GD')
    ens2 = []
    psi2 = psi
    for i in range(20):
        psi2, en = gd.gradient_descent (psi2, H, gamma=2e-7)
        ens2.append(en)
        #print(en)
    #plt.figure()
    #plt.plot(range(len(enss)), enss, marker='o')

    # Plot
    #mps = npmps.normalize_by_integral (psi2, x1, x2)
    #ptut.plot_1D (mps, x1, x2, ax=ax, label='GD')

    #en0 = sho.exact_energy(0)
    #en_errs = [en-en0 for en in ens0]
    #plt.figure()
    #plt.plot (range(len(ens0)), en_errs, marker='o')
    #plt.yscale('log')


    psi0 = npmps.normalize_by_integral (psi0, x1, x2)
    psi1 = npmps.normalize_by_integral (psi1, x1, x2)
    psi2 = npmps.normalize_by_integral (psi2, x1, x2)

    return psi0, ens0, psi1, ens1, psi2, ens2

if __name__ == '__main__':
    N = 4
    x1,x2 = -5,5

    fig, ax = plt.subplots()
    sho.plot_GS_exact(x1,x2,ax,ls='--',label='Exact')

    H = sho.make_H (N, x1, x2)
    psi = npmps.random_MPS (N,2,2)
    psi = npmps.compress_MPS (psi, cutoff=1e-12)
    psi = npmps.to_canonical_form (psi, 0)
    psi[0] /= np.linalg.norm(psi[0])

    psi0, ens0, psi1, ens1, psi2, ens2 = find_ground_state(H, psi)
    ptut.plot_1D (psi0, x1, x2, ax=ax, marker='o',label='DMRG')
    ptut.plot_1D (psi1, x1, x2, ax=ax, label='TDVP')
    ptut.plot_1D (psi2, x1, x2, ax=ax, label='GD')

    for i in range(1,4):
        H = sho.make_H (N+i, x1, x2)
        psi = copy.copy(psi0)
        psi = qtt.grow_site_1D (psi)
        psi0, ens0, psi1, ens1, psi2, ens2 = find_ground_state(H, psi)
        ptut.plot_1D (psi0, x1, x2, ax=ax, label='DMRG N='+str(N+i))
        ptut.plot_1D (psi1, x1, x2, ax=ax, label='TDVP N='+str(N+i))
        #ptut.plot_1D (psi2, x1, x2, ax=ax, label='GD N='+str(N+i))

    ax.legend()
    plt.show()
