import numpy as np

def make_rd(G,T,R0=None,d0=None):
    G = G.astype("float")
    T = T.astype("float")
    x = len(G[0,0,:])
    if R0 is None or d0 is None:
        if R0 is not None or d0 is not None:
            raise ValueError("R0 and d0 must both be supplied.")
    if R0 is None:
        R,d = finalize_row(G,T,0)
        start = 1
        for p in range(start,x):
            R,d = eliminate_row(R,d,p,p,x)
            R,d = finalize_row(R,d,p)
    else:
        R = G
        d = T
        R[:,:-1,:-1] = R0[None,:,:]
        d[:,:,:-1] = d0[:,None,:]
        R,d = finalize_column(R,d)
        start = len(R0[0,:])
    return R,d

def eliminate_row(G,T,k,col,m):
    R = G[:]
    d = T[:]
    Gr = np.array([G[:,k-1,col:m]]).transpose([1,0,2])
    GrT = Gr.transpose([0,2,1])
    d[:,:,col:m,0] = T[:,:,col:m,0] - T[:,:,k-1] * GrT[:,:,0]
    R[:,col:m,col:m] = G[:,col:m,col:m] - np.triu(np.matmul(GrT,Gr))
    return R,d

def finalize_column(R,d):
    for p in range(0,len(R[0,:])-1):
        R[:,p,-1] = R[:,p,-1] / R[0,p,p]
        R[:,p+1,-1] = R[:,p+1,-1] - np.sum(R[:,:p+1,p+1] * R[:,:p+1,-1], axis=1)
        d[:,:,-1,:] = d[:,:,-1,:] - d[:,:,p,:] * R[None,:,p,-1,None]
    R[:,-1,-1] = R[:,-1,-1] ** 0.5
    d[:,:,-1,:] = d[:,:,-1,:] / R[None,:,-1,-1,None]
    return R,d

def finalize_row(G,T,k):
    R = G[:]
    d = T[:]
    w = np.abs(G[:,k,k])**.5
    R[:,k,:] = (G[:,k,:]/w[:,None])
    d[:,:,k] = (T[:,:,k,:]/w[None,:,None])
    return R,d

