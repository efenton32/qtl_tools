import numpy as np

def make_rd(G,T):
    G = G.astype('double')
    T = T.astype('double')
    R,d = finalize_row(G,T,0)
    x = len(G[0,0,:])
    for p in range(1,x):
        R,d = eliminate_row(R,d,p,p,x)
        R,d = finalize_row(R,d,p)
    return R,d

def eliminate_row(G,T,k,col,m):
    R = G[:]
    d = T[:]
    Gr = np.array([G[:,k-1,col:m]]).transpose([1,0,2])
    GrT = Gr.transpose([0,2,1])
    x = T[:,k-1] * GrT[:,:,0]
    d[:,col:m,0] = T[:,col:m,0] - T[:,k-1] * GrT[:,:,0]
    R[:,col:m,col:m] = G[:,col:m,col:m] - np.triu(np.matmul(GrT,Gr))
    return R,d

def finalize_row(G,T,k):
    R = G[:]
    d = T[:]
    w = np.abs(G[:,k,k])**.5
    R[:,k,:] = (G[:,k,:].T/w).T
    d[:,k] = (T[:,k,:].T/w).T
    return R,d
