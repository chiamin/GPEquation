import os, sys 
sys.path.append('/home/chiamin/project/2023/qtt/JhengWei/INSTALL/xfac/build/python/')
import xfacpy

import numpy as np
import cmath

import time
import matplotlib.pyplot as plt
from matplotlib import cm
sys.path.append('/home/chiamin/project/2023/qtt/code/new/tools/')
import plot_utility as pltut

def fitfun_xy (x, y):
    f = np.sqrt(1/np.pi)*(x+1j*y)*np.exp(-(x**2+y**2)/2)*np.exp(2*np.pi*1j*np.random.rand())
    return f

def inds_to_num (inds):
    bstr = ''
    for i in inds:
        bstr += str(i)
    return pltut.bin_to_dec (bstr, dx, shift)

def fitfun (inds):
    N = len(inds)
    xinds, yinds = inds[:N//2], inds[N//2:]
    x = inds_to_num (xinds)
    y = inds_to_num (yinds)
    return fitfun_xy (x, y)

def xfac_to_npmps (mpsX, nsite):
  mps = [None for i in range(nsite)]
  for it in range(nsite):
    mps[it] = mpsX.get(it)
  return mps 

def tci (fun, length, phys_dim, maxdim, tol=1e-12, cplx=True, chk=False):
  fun.__name__
  incD = 2

  pm = xfacpy.TensorCI2Param()
  pm.pivot1 = [0]*int(length)

  pm.reltol = 1e-20
  pm.bondDim = 2 

  if (cplx):
    tci = xfacpy.TensorCI2_complex(fun, [phys_dim]*length, pm) 
  else:
    tci = xfacpy.TensorCI2(fun, [phys_dim]*length, pm) 

  it = 0 
  while (tci.param.bondDim < maxdim):
    t0 = time.time()
    tci.iterate(2,2)
    err0 = tci.pivotError[0]
    err1 = tci.pivotError[-1]
    print("tci: {0:10}| {1:5d} {2:20.3e} {3:20.3e} {4:20.3e} {5:20.2e}".
         format(fun.__name__, tci.param.bondDim, err0, err1, err1/err0, time.time()-t0), flush=True)
    if (err1/err0 < tol):
      break
    tci.param.bondDim = tci.param.bondDim + incD

  if (chk):
    print("tci.trueError = ", tci.trueError())

  return xfac_to_npmps(tci.tt, length) 

if __name__ == '__main__':
    N = 6
    x1, x2 = -6, 6
    Ndx = 2**N-1
    dx = (x2-x1)/Ndx
    shift = x1

    seed = 13
    np.random.seed(seed)

    xsite = ysite = N
    x_arr = np.linspace(x1, x2,2**(int(xsite)))
    xi2_arr= x_arr**2
    y_arr = np.linspace(x1, x2,2**(int(ysite)))
    yi2_arr= y_arr**2
    x, y = np.meshgrid(x_arr, y_arr)
    def trial_func(origx,origy):
        seed = 13
        np.random.seed(seed)
        x, y = np.meshgrid(origx, origy)
        #trial_func = x**2+y**2+1
        trial_func = np.sqrt(1/np.pi)*(x+1j*y)*np.exp(-(x**2+y**2)/2)*np.exp(2*np.pi*1j*np.random.random((len(x),len(y))))
        return trial_func
    #print(norm_arr(trial_func(x_arr,y_arr),x_arr,y_arr))
    #z = trial_func(x_arr,y_arr)/norm_arr(trial_func(x_arr,y_arr),x_arr,y_arr)
    z = trial_func(x_arr,y_arr)

    def absSqr (a): return abs(a)**2
    absSqr = np.vectorize(absSqr)

    fig,ax = plt.subplots(subplot_kw={"projection": "3d"})
    z = absSqr(z)
    surf = ax.plot_surface (x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    ff = np.vectorize (fitfun_xy)
    zz = ff (x, y)
    zz = absSqr(zz)
    fig,ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface (x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    mps = tci (fitfun, 2*N, 2, maxdim=30)
    pltut.plot_2D (mps, x1, x2, func=absSqr)

    plt.show()

