import sys
sys.path.append('/home/chiamin/project/2023/qtt/code/new/tools/')
import numpy as np
import npmps
import numpy_dmrg as dmrg
import matplotlib.pyplot as plt
import gradient_descent as gd

def get_TFIM_MPO_tensors (hz, hx):
    # Define operators
    I = np.array([[1.,0.],[0.,1.]])
    Sx = np.array([[0,1],[1,0]])
    Sz = np.array([[1,0],[0,-1]])

    # Define MPO tensor
    dM = 3                                      # MPO bond dimension
    M = np.zeros((dM, 2, 2, dM))
    M[0,:,:,0] = I
    M[2,:,:,2] = I
    M[1,:,:,0] = Sx
    M[2,:,:,1] = -Sx
    M[2,:,:,0] = -hz*Sz - hx*Sx

    L = np.array([0,0,1])
    R = np.array([1,0,0])
    return M, L, R

def get_TFIM_MPO (hz, hx):
    M, L, R = get_TFIM_MPO_tensors (hz, hx)
    H = [M for i in range(Nsites)]
    H = npmps.absort_LR (H, L, R)
    return H

if __name__ == '__main__':
    Nsites = 14       # Number of sites
    hz = 1          # Transverse field
    hx = 0         # Longitudinal field

    # Get initial state MPS
    psi0 = npmps.random_MPS (Nsites, 2, vdim=10)
    psi0 = npmps.to_canonical_form (psi0, 0)
    psi0[0] /= np.linalg.norm(psi0[0])

    # Define the bond dimensions for the sweeps
    maxdims = [8,16,32,64,128]
    cutoff = 1e-12

    # Initialiaze MPO
    H = get_TFIM_MPO (hz, hx)

    # Bond dimensions
    maxdims = [2]*2 + [4]*2 + [8]*2 + [16]*2 + [32]*2 + [64]*2 + [128]*2
    psi, ens, terrs = dmrg.dmrg (1, psi0, H, maxdims, cutoff=1e-12, krylovDim=4, verbose=False)
    for i,j in zip(ens,terrs): print(i,j)
    #plt.plot (terrs,ens,marker='o')
    #plt.show()

    print('TDVP')
    dt = 0.2
    maxdims = [2]*100 + [4]*2 + [8]*2 + [16]*2 + [32]*2 + [64]*2 + [128]*2
    psi3, ens3, terrs3 = dmrg.tdvp (1, psi0, H, dt, maxdims, cutoff=1e-12, krylovDim=4, verbose=False)
    for i,j in zip(ens3,terrs3): print(i,j)

    print('GD')
    enss = []
    psi2 = psi0
    for i in range(20):
        psi2, en = gd.gradient_descent (psi2, H, gamma=0.5)
        print(en)
        enss.append(en)
    plt.plot(range(len(enss)), enss, marker='o')
    plt.show()
