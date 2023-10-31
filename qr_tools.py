import numpy as np

def make_rd(G,T,R0=None,d0=None):
    G = G.astype("float")
    T = T.astype("float")
    x = G.shape[2]
    if (R0 is None and d0 is not None) or (d0 is None and R0 is not None):
        raise ValueError("R0 and d0 must both be supplied.")
    if R0 is None:
        R,d = finalize_row(G,T,0)
        start = 1
        for p in range(start,x):
            R,d = eliminate_row(R,d,p,p,x)
            R,d = finalize_row(R,d,p)
    else:
        R = G.copy()
        d = T.copy()
        n = R.shape[2] - R0.shape[2]
        k = R0.shape[2]
        R[:,:-n,:-n] = R0[None,:,:]
        d[:,:,:-n] = d0[:,None,:]
        for i in range(k+1,k+n+1):
            R[:,:i,:i],d[:,:,:i] = finalize_column(R[:,:i,:i],d[:,:,:i])
    return R,d

def make_rd_compact(G_c,T_c,R0,d0):
    x = G_c.shape[1]
    k = R0.shape[2]
    n = T_c.shape[2]
    R_c,d_c = finalize_column_compact(R0, d0, G_c, T_c)
    return R_c,d_c

def finalize_column(R,d):
    for p in range(R.shape[1] - 1):
        R[:,p,-1] = R[:,p,-1] / R[:,p,p]
        R[:,p+1,-1] = R[:,p+1,-1] - np.sum(R[:,:p+1,p+1] * R[:,:p+1,-1], axis=1)
        d[:,:,-1,:] = d[:,:,-1,:] - d[:,:,p,:] * R[None,:,p,-1,None]
    R[:,-1,-1] = R[:,-1,-1] ** 0.5
    d[:,:,-1,:] = d[:,:,-1,:] / R[None,:,-1,-1,None]
    return R,d

def finalize_column_compact(R0,d0,G_c,T_c):
    p = -1
    for p in range(R0.shape[1] - 1):
        G_c[:,p,0] = G_c[:,p,0] / R0[:,p,p]
        G_c[:,p+1,0] = G_c[:,p+1,0] - np.sum(R0[:,:p+1,p+1] * G_c[:,:p+1,0], axis=1)
        T_c[:,:,0,:] = T_c[:,:,0,:] - d0[:,None,p,:] * G_c[None,:,p,0,None]
    G_c[:,p+1,0] = G_c[:,p+1,0] / R0[:,p+1,p+1]
    G_c[:,p+2,0] = G_c[:,p+2,0] - np.sum(G_c[:,:p+2,0] * G_c[:,:p+2,0], axis=1)
    T_c[:,:,0,:] = T_c[:,:,0,:] - d0[:,None,p+1,:] * G_c[None,:,p+1,0,None]

    G_c[:,-1,0] = G_c[:,-1,0] ** 0.5
    T_c[:,:,0,:] = T_c[:,:,0,:] / G_c[None,:,-1,-1,None]
    return G_c,T_c

def eliminate_row(G,T,k,col,m):
    R = G[:]
    d = T[:]
    Gr = np.array([G[:,k-1,col:m]]).transpose([1,0,2])
    GrT = Gr.transpose([0,2,1])
    d[:,:,col:m,0] = T[:,:,col:m,0] - T[:,:,k-1] * GrT[:,:,0]
    R[:,col:m,col:m] = G[:,col:m,col:m] - np.triu(np.matmul(GrT,Gr))
    return R,d

def finalize_row(G,T,k):
    R = G[:]
    d = T[:]
    w = np.abs(G[:,k,k])**.5
    R[:,k,:] = (G[:,k,:]/w[:,None])
    d[:,:,k] = (T[:,:,k,:]/w[None,:,None])
    return R,d

