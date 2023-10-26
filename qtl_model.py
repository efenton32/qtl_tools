import numpy as np
import time
import numpy.linalg as linalg
import qr_tools as qr
import pandas as pd
import matplotlib.pyplot as plt
import os
import json

class QTL_model:
    def __init__(self, pheno, geno, names = None, samps = None,
                 chr_edges = [0, 995, 4730, 5288, 9323, 11182, 12470, 16401, 18039, 20117, 23091, 26330, 
                              30640, 33585, 35384, 39673, 41594]
    ):
        """
        Initializes a QTL model

        Parameters:
            pheno [numpy array]: Array of phenotypic data. Shape (# envs, # segregants)
            geno [numpy array]: Array of genotype data. Shape (# segregants, # SNPs)
            names [list of str]: Environment names, optional
            chr_edges [list of int]: SNP indices that define chromosome boundaries
        """
        np.set_printoptions(suppress=True)
        t = time.time()
        if type(geno) == str:
            geno = np.load(geno)
            geno = geno[samps,:]
        if geno.shape[0] != pheno.shape[1]:
            raise ValueError("First dim of genotypes and 2nd dim of phenotypes must match")
        self.pheno = pheno[:,:,None]
        self.N = geno.shape[0]
        self.L = geno.shape[1] + 1
        self.num_env = self.pheno.shape[0]
        self.chr_edges = chr_edges
        
        # Make genotypes between -1 and 1 if they're given 0~1
        if np.min(geno) > 0:
            geno = geno * 2 - 1
        # Add a locus of all 1's for all segregants at the end of the genotype matrix
        # This effectively acts as the "offset" locus
        self.geno = np.hstack( [geno, np.ones((self.N, 1))] )
        del geno

        # Stores the magnitudes of the genotype vectors
        self.dots = np.sum(self.geno*self.geno, axis = 0)[:,None]

        if names is not None:
            self.names = names
        else:
            # Placeholder names if env names are not given
            self.names = np.arange(self.num_env).astype("str")
        print("Initialize: {}".format(round(time.time() - t)))

    def run(self, num_qtl=20, refit=None, b_plots=False, plot_path=None, b_save_itm=False, outfile=None, statfile=None,
            corr_cutoff = 0):
        """
        Perform the forward-search algorithm

        Parameters:
            num_qtl [int]: Number of QTLs to find, defaults to 20
            b_plots [bool]: Boolean to indicate whether to plot and save each step
            path [str]: Path to save plots to, only used if b_plots = True
            b_save_itm [bool]: Boolean to save intermediate steps. If True, saves results every 10 steps.
            outfile [str]: File to save intermdiate steps to. Unused if b_save_itm = False.
        """
        # Make the directory for saving plots if it doesn't already exist
        if b_plots:
            try:
                os.mkdir(plot_path)
            except FileExistsError:
                pass
        t = time.time()
        # S0 represents the loss function, or the squared residuals. With a blank model, the residuals
        # are just the phenotypes squared
        self.corr_cutoff = corr_cutoff
        self.S0 = np.sum(self.pheno**2, axis=1)
        self.b_plots = b_plots
        self.plot_path = plot_path
        self.b_save_itm = b_save_itm
        self.outfile = outfile
        self.num_qtl = num_qtl
        self.refit = refit
        self.statfile = statfile
        # Stores the summed squared residuals in every environment at every step
        self.S = np.zeros((self.num_qtl + 1, self.num_env))
        # Stores the summed squared residuals, summed across environments at every step. This is the
        # loss we are trying to minimize
        self.S_tot = np.zeros((self.num_qtl + 1, 1))

        # The summed squared residuals are just the phenotypes squared for an empty model
        self.S[0,:,None] = np.sum( (self.pheno - np.mean(self.pheno, axis=1, keepdims=True))**2, axis = 1)
        self.S_tot[0] = np.sum(self.S[0], axis=0)
        self.pos = np.zeros((self.num_qtl + 1, 1)).astype("int")

        # G holds the dot products of loci in the model with each other all potential new loci to add 
        # to the model
        self.G = np.zeros((1, 1, 1))
        # T holds the dot products of the phenotype vectors with loci in the model and all potential
        # new loci to add to the model
        self.T = np.zeros((self.num_env, 1, 1))
        
        # Tracks the step of forward search we're currently on. Fitting the offset is considered step 0
        self.step = 0
        # add_first() adds the first "QTL". This is ALWAYS at the last locus and represents the offset.
        self.add_first()
        print("Add first: {}".format(round(time.time() - t)))
        # Add qtls to the modeul until we've added the specified number.
        while self.step < self.num_qtl + 1:
            t = time.time()
            self.qtl_step()
            print()
            print("Step {}: {}".format(self.step, round(time.time() - t)))
            self.step += 1
        self.step -= 1
        self.generate_s_vals()
        self.step += 1

    def add_first(self):
        """
        Adds the first 'QTL' to the model. Note that this is not a true QTL, but represents the offset,
        eg. centers the phenotype data to have 0 mean. This is always at the last position in the genome.
        """
        self.S_tot[self.step] = self.S_tot[0]
        self.S[self.step] = self.S[0]
        self.pos[0] = self.L - 1

        self.T[:,0,0] = np.sum(self.geno[:,self.pos[0]]*self.pheno[:,:,0].T, axis = 0)
        self.G[0,0,0] = np.sum(self.geno[:,self.pos[0]]*self.geno[:,self.pos[0]])
        self.step += 1

    def qtl_step(self):
        self.add_qtl()
        if self.refit is not None:
            if self.step % self.refit == 0:
                self.refit_pos()
        # if self.b_save_itm and self.step % 10 == 0:
        #     self.generate_s_vals()
        #     save_name = self.outfile.split(".")[0] + "_step_{}.csv".format(self.step+1)
        #     self.dump_qtls(self.outfile)
        if self.statfile is not None:
            self.generate_s_vals()
            save_name = self.outfile.split(".")[0] + "_step_{}.csv".format(self.step)
            self.dump_qtls(save_name)
            self.dump_stats()

    def dump_stats(self):
        with open(self.statfile, "a") as sf:
            self.generate_s_vals()
            sf.write(json.dumps(
                {"step": self.step, "pos": self.pos[:self.step+1,0].tolist(), "loss": float(self.S_tot[self.step,0])}
            ))
            sf.write("\n")

    def add_qtl(self):
        m = self.step + 1
        t = time.time()
        G = np.zeros((self.L, m, m))
        G[:,0:-1,0:-1] = self.G[:,:,:]
        G[:,-1,-1] = self.dots[:,0]
        G[:,0:-1,-1] = np.matmul(self.geno[:,:].T, self.geno[:,self.pos[:m-1,0]])

        T = np.zeros((self.num_env, self.L, m, 1))
        T[:,:,0:-1,0] = self.T[:,:,:].transpose((0,2,1))
        T[:,:,-1,:] = np.matmul(self.geno[:,:].T,self.pheno)
        if self.step == 1:
            R,d = qr.make_rd(G,T)
        else:
            R,d = qr.make_rd(G,T,self.R,self.d)
        S = self.S0 - np.sum(d**2, axis=2)[:,:,0]
        S = np.array(S)
        S_tot = np.sum(S, axis = 0)
        S_tot[R[:,-1,-1] < self.corr_cutoff] = np.inf

        #### Plot ####
        if self.b_plots:
            plt.figure()
            plt.plot(self.S_tot[self.step-1] - S_tot, label="Total benefit", color="black")
            for i in range(self.num_env):
                plt.plot(self.S[self.step-1][i] - S[i], label="Benefit " + self.names[i])
            plt.xlabel("SNP index")
            plt.ylabel("Cost function benefit")
            plt.tight_layout()
            plt.savefig(self.plot_path + str(m-1) + ".pdf")
            plt.close()

        #### Update variables for next iteration ####
        S_tot[self.pos[:m-1]] = np.inf
        p = np.nanargmin(S_tot)

        print("Step: {}".format(self.step))
        print("R: {}".format(R[p,-1,-1]))
        print("p: {}".format(p))
        self.T = T[:,p]
        self.G = G[None,p,:,:]
        self.d = d[:,p]
        self.R = R[None,p,:,:]
        del G, T, R, d

        self.pos[m-1] = p
        self.S_tot[self.step] = S_tot[p]
        self.S[self.step] = S[:,p]

    def refit_pos(self):
        for q in range(1, self.step + 1):
            pos_sorted = np.sort(self.pos[1:self.step+1,0])
            chrom = np.searchsorted(self.chr_edges, self.pos[q], "right")[0] - 1
            edge_low = self.chr_edges[chrom]
            edge_high = self.chr_edges[chrom+1] - 1
            gap = np.where(pos_sorted == self.pos[q,0])[0]
            if gap == 0:
                bottom = edge_low
                top = int(min(edge_high, pos_sorted[gap+1]-1))
            elif gap+1 == len(pos_sorted):
                bottom = int(max(edge_low, pos_sorted[gap-1]+1))
                top = edge_high
            else:
                bottom = int(max(edge_low, pos_sorted[gap-1]+1))
                top = int(min(edge_high, pos_sorted[gap+1]-1))
            G = np.zeros([top-bottom+1, self.step+1, self.step+1])
            T = np.zeros([self.num_env, top-bottom+1, self.step+1, 1])
            G[:,:,:] = self.G
            G[:,:q,q] = np.dot(self.geno[:,bottom:top+1].T, self.geno[:,self.pos[:q,0]])
            G[:,q,q] = self.dots[bottom:top+1,0]
            G[:,q,q+1:] = np.dot(self.geno[:,bottom:top+1].T, self.geno[:,self.pos[q+1:self.step+1,0]])
            
            T[:,:,:,:] = self.T[:,None,:,:]
            T[:,:,q,:] = np.matmul(self.geno[:,bottom:top+1].T, self.pheno)
            R,d = qr.make_rd(G, T, R0 = self.R[:,:q,:q], d0 = self.d[:,:q,:])
            S = self.S0 - np.sum(d**2, axis=2)[:,:,0]
            S = np.array(S)
            S_tot = np.sum(S, axis = 0)
            S_tot[np.min(np.diagonal(R, axis1=1, axis2=2),axis=1) < self.corr_cutoff] = np.inf
            p = np.nanargmin(S_tot)

            #### Plot ####
            if self.b_plots:
                plt.figure()
                a = max(S_tot)
                b = max(a - S_tot)
                plt.plot(a - S_tot, label="Total benefit", color="black")
                plt.plot([self.pos[q,0]-bottom, self.pos[q,0]-bottom], [0, b], "--", color="gray", label="Original pos")
                plt.plot([p, p], [0, b], "--", color="orchid", label = "New pos")
                for i in range(self.num_env):
                    a = max(S[i])
                    plt.plot(a-S[i], label="Benefit " + self.names[i])
                plt.xlabel("SNP index")
                plt.ylabel("Cost function benefit")
                plt.tight_layout()
                plt.savefig(self.plot_path + str(self.step) + "q_{}.pdf".format(q))
                plt.close()

            if p + bottom != self.pos[q,0]:
                print("Step = {}".format(self.step))
                print("QTL {} moved from {} to {}".format(q, self.pos[q,0], p + bottom))
                print("R: {}".format(R[p,-1,-1]))
            self.T = T[:,p]
            self.G = G[None,p,:,:]
            self.d = d[:,p]
            self.R = R[None,p,:,:]
            del G, T, R, d

            self.pos[q,0] = p + bottom
            self.S_tot[self.step] = S_tot[p]
            self.S[self.step] = S[:,p]

    def dump_qtls(self, outfile):
        columns = [" " for i in range(len(self.names)+1)]
        columns[1:] = self.names
        columns[0] = "pos"
        df = pd.DataFrame(self.s_vals, index=self.pos[:self.step+1,0], columns=self.names)
        df.to_csv(outfile, sep="\t")

    def generate_s_vals(self):
        self.s_vals = np.zeros((self.step+1, self.num_env))
        R_inv = linalg.inv(self.R)
        self.s_vals[:,:] = np.matmul(R_inv,self.d).transpose((1,0,2))[:,:,0]

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
            s_vals = np.matmul(R_inv, d+1)
            calc = []
            for i in range(self.elim_step):
                calc.append( np.matmul(self.geno[env][:, self.pos[self.step - 1][0:i+1, env]], s_vals[i][0:i+1])
                           + np.matmul(self.geno[env][:, self.pos[self.step - 1][i+2:, env]], s_vals[i][i+1:]) )

            calc = np.array(calc)
            S = linalg.norm(self.pheno[env] - calc, axis=1)
            p = np.nanargmin(S)

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


def validate(pheno, geno, qtls, return_pred=False):
    if np.min(geno) >= 0:
        geno = geno*2 - 1
    geno = np.hstack( [geno, np.ones((geno.shape[0],1))] )
    pred = np.matmul(geno[:,qtls.index], qtls.to_numpy())
    S = np.nansum((pred - pheno.T)**2, axis=0)
    S_tot = np.sum(S)
    if return_pred:
        return S, S_tot, pred.T
    else:
        return S, S_tot

