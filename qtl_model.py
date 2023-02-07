import numpy as np
import time
import numpy.linalg as linalg
import qr_tools as qr

import matplotlib.pyplot as plt

class QTL_model:

    def __init__(self, pheno, geno, names = None):
        """
        Initializes a QTL model

        Parameters:
            pheno [numpy array]: Array of phenotypic data. Shape (# envs, # lineages)
            geno [numpy array]: Array of genotype data. Shape (# lineages, # SNPs)
            names [list of str]: Environment names, optional
        """
        if geno.shape[0] != pheno.shape[1]:
            raise ValueError("First dim of genotypes and 2nd dim of phenotypes must match")
        self.pheno = pheno[:,:,None]
        self.N = geno.shape[0]
        self.L = geno.shape[1] + 1
        self.num_env = self.pheno.shape[0]
        
        if np.min(geno) > 0:
            # Make genotypes between -1 and 1 if they're given 0~1
            geno = geno * 2 - 1
        self.geno = np.hstack( [geno, np.ones((self.N, 1))] )
        del geno

        # Magnitudes of the genotype vectors
        self.dots = np.sum(self.geno*self.geno, axis = 0)[:,None]

        if names is not None:
            self.names = names
        else:
            # Placeholder names if env names are not given
            self.names = np.arange(self.num_env).astype('str')

    def run(self, num_qtl = 20, b_plots = False, path = None):
        """
        Perform the forward-search algorithm

        Parameters:
            num_qtl [int]: Number of QTLs to find, defaults to 20
            b_plts [bool]: Whether to plot and save each step
            path [str]: Path to save plots to, only used if b_plots = True

        """
        self.b_plots = b_plots
        self.path = path
        self.num_qtl = num_qtl
        # Stores the MSE in every environment at every step
        self.S = np.zeros((self.num_qtl + 1, self.num_env))
        # Stores the total MSE across environments at every step
        self.S_tot = np.zeros((self.num_qtl + 1, 1))

        self.S[0,:,None] = np.sum(self.pheno**2, axis = 1)**0.5
        self.S_tot[0] = sum(self.S[0]**2)**0.5
        self.pos = np.zeros((self.num_qtl + 1, 1)).astype("int")
        self.s_vals = np.zeros((self.num_qtl + 1, self.num_env))

        self.G = np.zeros((1, 1, 1))
        self.T = np.zeros((self.num_env, 1, 1))
        
        self.step = 0
        self.add_first()
        while self.step < self.num_qtl + 1:
            self.qtl_step()

    def add_first(self):
        """
        Adds the first 'QTL' to the model. Note that this is not a true QTL, but represents the offset,
        eg. centers the phenotype data to have 0 mean.
        """
        self.S_tot[self.step] = self.S_tot[0]
        self.S[self.step] = self.S[0]
        self.pos[0] = self.L - 1

        self.T[:,0,0] = np.sum(self.geno[:,self.pos[0]]*self.pheno[:,:,0].T, axis = 0)
        self.G[0,0,0] = np.sum(self.geno[:,self.pos[0]]*self.geno[:,self.pos[0]])
        self.step += 1

    def qtl_step(self):
        self.add_qtl()

    def add_qtl(self):
        m = self.step + 1
        print(self.step)
        print(m)
        print(self.S_tot)
        t = time.time()
        G = np.zeros((self.L, m, m))
        G[:,0:-1,0:-1] = self.G[:,:,:]
        G[:,-1,-1] = self.dots[:,0]
        G[:,0:-1,-1] = np.matmul(self.geno[:,:].T, self.geno[:,self.pos[:m-1,0]])

        T = np.zeros((self.num_env, self.L, m, 1))
        S = np.zeros((self.num_env, self.L))
        s_vals = np.zeros((self.num_env, self.L, m, 1))

        for i in range(self.num_env):
            t = time.time()
            T[i,:,0:-1,0] = self.T[i,:,0]
            T[i,:,-1,:] = np.matmul(self.geno[:,:].T,self.pheno[i])

            t = time.time()
            R,d = qr.make_rd(G,T[i])
            R_inv = linalg.inv(R)
            s_vals[i] = np.matmul(R_inv,d)
            del R, R_inv, d

            t = time.time()
            calc = np.matmul( self.geno[:,self.pos[:m-1,0]], s_vals[i,:,:-1,0].T ) \
                              + self.geno[:,:] * s_vals[None,i,:,-1,0]
            S[i] = linalg.norm(self.pheno[i] - calc, axis = 0)
            del calc

        S = np.array(S)
        S_tot = np.sum(S**2, axis = 0)**0.5

        #### Plot ####
        if self.b_plots:
            plt.figure()
            plt.plot(self.S_tot[self.step-1] - S_tot, label="Total benefit", color="black")
            for i in range(self.num_env):
                plt.plot(self.S[self.step-1][i] - S[i], label="Benefit " + self.names[i])
            plt.legend()
            # plt.ylim(0, )
            plt.ylabel("Cost function benefit")
            plt.tight_layout()
            plt.savefig(self.path + str(m-1) + ".pdf")

        #### Update variables for next iteration ####
        S_tot[self.pos[:m-1]] = np.inf
        p = np.argmin(S_tot)

        self.T = T[:,p]
        self.G = G[None,p,:,:]
        del G, T

        self.pos[m-1] = p
        s_vals = np.array(s_vals)
        self.s_vals[:m] = s_vals[:,p,:,0].T
        print(self.s_vals)
        print(self.pos)
        self.S_tot[self.step] = S_tot[p]
        self.S[self.step] = S[:,p]
        self.step += 1

    def back_eliminate(self):
        self.elim_step = self.step - 1
        while self.elim_step > 1:
            self.remove_qtl()

    def remove_qtl(self):
        self.pos.append(np.zeros([self.elim_step, self.num_env]).astype('int'))
        self.s_vals.append(np.zeros([self.elim_step, self.num_env]))
        for env in range(self.num_env):
            G = np.zeros([self.elim_step, self.elim_step, self.elim_step])
            T = np.zeros([self.elim_step, self.elim_step, 1])

            for i in range(self.elim_step):
                G[i, 0:i+1, 0:i+1] = self.G[env][0:i+1, 0:i+1]
                G[i, 0:i+1, i+1:] = self.G[env][0:i+1, i+2:]
                G[i, i+1:, i+1:] = self.G[env][i+2:, i+2:]

                T[i, 0:i+1] = self.T[env][0:i+1]
                T[i, i+1:] = self.T[env][i+2:]

            R,d = qr.make_rd(G, T)
            R_inv = linalg.inv(R)
            s_vals = np.matmul(R_inv, d)
            calc = []
            for i in range(self.elim_step):
                calc.append( np.matmul(self.geno[env][:, self.pos[self.step - 1][0:i+1, env]], s_vals[i][0:i+1])
                           + np.matmul(self.geno[env][:, self.pos[self.step - 1][i+2:, env]], s_vals[i][i+1:]) )

            calc = np.array(calc)
            S = linalg.norm(self.pheno[env] - calc, axis=1)
            p = np.argmin(S)

            self.G[env] = G[p]
            self.T[env] = T[p]
            self.pos[self.step][:, env] = np.hstack([self.pos[self.step - 1][0:p+1, env],
                                                    self.pos[self.step - 1][p+2:, env]])
            self.s_vals[self.step][:,env] = s_vals[p].ravel()

        self.elim_step -= 1
        self.step += 1


    def validate(self, v_pheno, v_geno):
        num_envs = len(v_pheno)
        num_models = len(self.s_vals)
        self.pred_err = np.zeros([num_envs, num_models])
        self.calc = [[] for i in range(num_envs)]

        for i in range(num_envs):
            N = len(v_pheno[i])
            v_geno[i] = np.hstack( [v_geno[i], np.ones([N,1])] )

        for env in range(num_envs):
            for model in range(num_models):
                self.calc[env].append(np.matmul( v_geno[env][:, self.pos[model][:,env]], self.s_vals[model][:,env] ))
                calc = np.matmul( v_geno[env][:, self.pos[model][:,env]], self.s_vals[model][:,env] )
                self.pred_err[env, model] = np.linalg.norm( v_pheno[env] - calc )


################################################################################

# Below are old functions that I've either incorporated already or plan to
# incorporate in the future

################################################################################

def add_first(geno,pheno,num_ind):
    num_env = pheno.shape[0]

    s_vals = np.dot(geno.T,pheno)/num_ind
    err = linalg.norm(pheno - s_vals.T*geno,axis = 0)
    pos = np.array(np.argmin(err))
    s_vals = np.array(s_vals[pos])
    err = linalg.norm(pheno[:,0] - geno[:,pos]*s_vals)
    return s_vals,err,pos

################################################################################

def refit(geno,X,G,pos,b,b_explained):
    N = len(geno[:,0])
    b_temp = np.matmul(geno[:,pos],X)
    b0 = b - b_explained + b_temp
    T = np.matmul(geno[:,pos].T,b0)

    R,d = qr.make_rd(np.array([G]),np.array([T]))
    R_inv = linalg.inv(R[0])
    X = np.matmul(R_inv,d[0])
    meas = np.matmul(geno[:,pos],X)
    # plt.figure()
    # plt.plot(b,b_explained,'.',label='old')
    b_explained = b_explained - b_temp + meas
    # plt.plot(b,b_explained,'.',label='new')
    # plt.legend()
    # plt.show()
    S = linalg.norm(b - b_explained)

    return X,S,T,b0,b_explained,meas

################################################################################

def qr_step(geno,X0,S0,T0,G0,pos0,b0):
    m = len(X0)+1
    N = len(geno[:,0])
    L = len(geno[0,:])

    G = np.zeros([L,m,m])
    T = np.zeros([L,m,1])
    pos = np.zeros([m],dtype='int')
    G[:,0:-1,0:-1] = G0[:,:]
    G[:,-1,-1] = np.sum(geno*geno,axis=0)
    G[:,range(m-1),-1] = np.matmul(geno[:,:].T,geno[:,pos0])
    T[:,range(m-1),0] = T0[:,0]
    T[:,-1,:] = np.matmul(geno[:,:].T,b0)

    R,d = qr.make_rd(G,T)

    R_inv = linalg.inv(R)
    X = np.matmul(R_inv,d)

    meas = np.matmul(geno[:,pos0],X[:,0:-1,0].T) + geno[:,:]*X[:,-1,0]
    S = linalg.norm(b0 - meas,axis=0)
    S[pos0] = np.inf

    # plt.figure()
    # plt.plot(S**2/N)
    # plt.plot(pos0,S[pos0+1]**2/N,'.')
    # plt.show()


    p = np.argmin(S)
    X = X[p]
    S = S[p]
    T = T[p]
    G = G[p]

    pos[0:-1] = pos0
    pos[-1] = p
    return X,S,T,G,pos

################################################################################

def jitter(geno,X0,S0,T0,G0,pos,b):

    k = len(pos)
    index = 0
    counter = 1
    dir = 'r'
    N = len(geno[:,0])

    G0 = np.array([G0])
    T0 = np.array([T0])
    G = np.zeros(G0.shape)
    T = np.zeros(T0.shape)

    while counter < k:

        while dir == 'r':
            G[0,:,:] = G0[:,:]
            T[0,:] = T0[:]
            p_temp = pos[index]+1

            if pos[index]==len(geno[0,:])-1 or pos[index]==0 or p_temp in pos:
                X = X0
                S = np.inf
                T = T0
                G = G0

            else:
                A = geno[:,pos]
                thing = np.dot(geno[:,p_temp],A)

                G[0,0:index,index] = thing[0:index]
                G[0,index,index+1:] = thing[index+1:]
                T[0,index] = np.matmul(geno[:,p_temp],b)

                R,d = qr.make_rd(G,T)
                R_inv = linalg.inv(R)
                X = np.matmul(R_inv,d)[0]

                meas = np.dot(geno[:,pos[range(index)]],X[range(index)])
                meas = meas + np.array([geno[:,p_temp]*X[index]]).T
                meas = meas + np.dot(geno[:,pos[index+1:]],X[index+1:])

                S = linalg.norm(meas-b)

            if S < S0:
                X0 = X
                S0 = S
                T0 = T
                G0 = G
                counter = 0
                pos[index] = pos[index] + 1
                print('Position '+str(pos[index]-1)+' moved 1 right')
            else:
                dir = 'l'

        while dir == 'l':
            G[:,:] = G0[:,:]
            T[:] = T0[:]
            p_temp = pos[index]-1

            if pos[index]==len(geno[0,:])-1 or pos[index]==0 or p_temp in pos:
                X = X0
                S = np.inf
                T = T0
                G = G0

            else:
                A = geno[:,pos]
                thing = np.dot(geno[:,p_temp],A)

                G[0,0:index,index] = thing[0:index]
                G[0,index,index+1:] = thing[index+1:]
                T[0,index] = np.matmul(geno[:,p_temp],b)

                R,d = qr.make_rd(G,T)
                X = np.matmul(R,d)[0]

                meas = np.dot(geno[:,pos[range(index)]],X[range(index)])
                meas = meas + np.array([geno[:,p_temp]*X[index]]).T
                meas = meas + np.dot(geno[:,pos[index+1:]],X[index+1:])

                S = linalg.norm(meas-b)

            if S < S0:
                X0 = X
                S0 = S
                T0 = T
                G0 = G
                counter = 0
                pos[index] = pos[index] - 1
                print('Position '+str(pos[index]+1)+' moved 1 left')
            else:
                dir = 'r'
                counter+=1

    return X0,S0,T0[0],G0[0],pos
