import numpy as np
import tt

def TT_product(a_cores, b_cores):
  """
  performs the TT chain of product between the two set of cores:
  -- a1 -- a2 -- a3 -- ...
     |     |     |
  -- b1 -- b2 -- b3 -- ...
  """
  for i,(c1,c2) in enumerate(zip(a_cores,b_cores)):
    if i == 0:
      res = np.tensordot(c1,c2,(1,1)).transpose([0,2,1,3])
    else:
      res = np.tensordot(res,c1,(2,0))
      res = np.tensordot(res,c2,((2,3),(0,1)))
  return res

def _rightorth(a, b):
    """
    right orthonormalisation of core a. After this we have
    np.tensordot(a,a,axes=((0,1),(0,1))) = I
    while
    np.tensordot(a,b,axes=(2,0)
    remains unchanged
    """
    adim = np.array(a.shape)
    bdim = np.array(b.shape)
    #print(adim, bdim)
    cr = np.reshape(a, (adim[0]*adim[1], adim[2]), order='F')
    cr2 = np.reshape(b, (bdim[0],bdim[1]*bdim[2]), order='F')
    [cr, R] = np.linalg.qr(cr)
    #print(R.shape, cr2.shape, cr.shape)
    cr2 = np.dot(R, cr2)

    adim[2] = cr.shape[1]
    bdim[0] = cr2.shape[0]
    a = np.reshape(cr, adim, order='F')
    b = np.reshape(cr2, bdim, order='F')
    return a,b

def _leftorth(a,b):
    """
    left orthonormalisation of core a. After this we have
    np.tensordot(b,b,axes=((1,2),(1,2))) = I
    while
    np.tensordot(a,b,axes=(2,0)
    remains unchanged
    """

    adim = np.asarray(a.shape)
    bdim = np.asarray(b.shape)
    cr = np.reshape(b, (bdim[0],bdim[1]*bdim[2]),order='F').T
    [cr, R] = np.linalg.qr(cr)
    cr2 = np.reshape(a, (adim[0]*adim[1],adim[2]),order='F').T
    cr2 = np.dot(R, cr2)

    adim[2] = cr2.T.shape[1]
    bdim[0] = cr.T.shape[0]

    a = np.reshape(cr2.T, adim, order='F')
    b = np.reshape(cr.T, bdim, order='F')

    return a,b

def _orthcores(a,dir='right'):
    """
    right (resp. left) orthonormalize all cores of a except for
    the last (resp. first) one
    """
    if isinstance(a,list):
        d = len(a)
        ry = [X.shape[0] for X in a] + [1]
        L = a
        #a = tt.vector.from_list(L)
    elif isinstance(a,tt.vector):
        d = a.d
        ry = a.r
        L = tt.vector.to_list(a)
    else:
        raise NotImplementedError()

    if dir=='right':
        for i in range(d - 1):
            L[i:i+2] = _rightorth(L[i], L[i + 1])
            ry[i + 1] = L[i].shape[2]
    elif dir=='left':
        for i in range(d-1,0,-1):
            L[i-1:i+1] = _leftorth(L[i-1],L[i])
            ry[i] = L[i-1].shape[2]

    return tt.vector.from_list(L) if isinstance(a,tt.vector) else L



def TT_factorisation_pinv(cores,n_row_modes):
    """
    assuming cores are the cores of the TT decomposition of H and n_row modes is the number
    of modes corresponding to 'prefixes' (i.e. l for H^{(2l)}), this returns the cores
    of the TT-decompositions of the pseudo-inverses of P and S, where P and S are such that
    H = PS
    """
    #print(len(cores))
    for i in range(n_row_modes):
        cores[i:i+2] = _rightorth(cores[i],cores[i+1])
    S_cores = _orthcores(cores[n_row_modes:], dir='left')
    P_cores = cores[:n_row_modes]

    c = S_cores[0]
    #print((c.reshape((c.shape[0],np.prod(c.shape[1:])))).shape)
    U,s,V = np.linalg.svd(c.reshape((c.shape[0],np.prod(c.shape[1:])),order='F'),full_matrices=False)
    P_cores[-1] = np.tensordot(P_cores[-1],U,axes=(2,0))
    S_cores[0] = (np.diag(1./s).dot(V)).reshape(c.shape,order='F')

    return P_cores,S_cores

def TT_spectral_learning(H_2l_cores, H_2l1_cores, H_l_cores):
    l = len(H_l_cores)
    P_cores, S_cores = TT_factorisation_pinv(H_2l_cores,l)

    alpha = TT_product(H_l_cores,S_cores)
    A_left = TT_product(H_2l1_cores[:l],P_cores)
    A_right = TT_product(H_2l1_cores[l+1:],S_cores)
    A = np.tensordot(A_left,H_2l1_cores[l],(2,0))
    A = np.tensordot(A,A_right,(4,0))

    # compute  omega
    omega = TT_product(H_l_cores,P_cores)

    return alpha.squeeze(), A.squeeze(), omega.squeeze()
