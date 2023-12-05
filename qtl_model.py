import numpy as np
import time
import numpy.linalg as linalg
import qr_tools as qr
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from itertools import combinations

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

    def run(self, num_qtl=400, refit=None, b_plots=False, plot_path=None, b_save_itm=False, outfile=None, statfile=None,
            corr_cutoff=0):
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
        self.epistatic = False
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
        self.dump_qtls(self.outfile)
        self.step += 1

    def run_epistasis(self, num_e_qtl=400, b_plots=False, plot_path=None, b_save_itm=False,
                      outfile=None, statfile=None):
        """
        Finds epistatic loci from a fit additive model. Only considers interactions between additive QTLs, not
        with other SNPs.
        """
        if b_plots:
            try:
                os.mkdir(plot_path)
            except FileExistsError:
                pass
        self.b_plots = b_plots
        self.plot_path = plot_path
        self.b_save_itm = b_save_itm
        self.outfile = outfile
        self.num_e_qtl = num_e_qtl
        self.statfile = statfile
        self.refit = None
        self.corr_cutoff = 0
        self.epistatic = True

        self.S0 = np.sum(self.pheno**2, axis=1)
        self.S = np.zeros((self.num_e_qtl + self.step + 1, self.num_env))
        self.S_tot = np.zeros((self.num_e_qtl + self.step + 1, 1))

        self.S[:self.step+1,:] = (self.S0[:,None,:] - np.cumsum(self.d**2, axis=1))[:,:,0].transpose((1,0))
        self.S_tot = np.sum(self.S, axis=1)[:,None]

        self.add_pos = self.pos.copy()
        self.e_loci = list(combinations(self.pos[1:], 2))
        self.geno = np.hstack( [self.geno[:,self.pos], np.prod(self.geno[:,self.e_loci], axis = 2)])
        self.L = self.geno.shape[1]
        self.pos = np.zeros([len(self.pos) + self.num_e_qtl, 1])
        self.pos[:self.step + 1,0] = np.arange(self.step + 1)
        self.pos = self.pos.astype("int")
        
        self.dots = np.sum(self.geno*self.geno, axis = 0)[:,None]
        self.step += 1

        while self.step < self.num_e_qtl + len(self.add_pos):
            t = time.time()
            self.qtl_step()
            print()
            print("Epistasis step {}: {}".format(self.step, round(time.time() - t)))
            self.step += 1
        self.step -= 1
        self.generate_s_vals()
        self.dump_qtls(self.outfile)
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
        self.R = self.G ** 0.5
        self.d = self.T / self.R
        self.step += 1

    def resume(self, num_qtl=20, refit=None, b_plots=False, plot_path=None, b_save_itm=False, outfile=None,
               statfile=None, corr_cutoff=0):

        t = time.time()

        self.corr_cutoff = corr_cutoff
        self.S0 = np.sum(self.pheno**2, axis=1)
        self.b_plots = b_plots
        self.plot_path = plot_path
        self.b_save_itm = b_save_itm
        self.outfile = outfile
        self.num_qtl = num_qtl
        self.refit = refit
        self.statfile = statfile
        self.epistatic = False

        self.S = np.zeros((self.num_qtl + 1, self.num_env))
        self.S_tot = np.zeros((self.num_qtl + 1, 1))

        self.S[:self.step+1,:] = (self.S0[:,None,:] - np.cumsum(self.d**2, axis=1))[:,:,0].transpose((1,0))
        self.S_tot = np.sum(self.S, axis=1)[:,None]
        x = self.pos.copy()
        self.pos = np.zeros((self.num_qtl + 1, 1)).astype("int")
        self.pos[:self.step+1,0] = x.copy()
        self.step += 1
        del x
        while self.step < self.num_qtl + 1:
            t = time.time()
            self.qtl_step()
            print()
            print("Step {}: {}".format(self.step, round(time.time() - t)))
            self.step += 1
        self.step -= 1
        self.generate_s_vals()
        self.step += 1
        
    def qtl_step(self):
        self.add_qtl()
        if self.refit is not None:
            if self.step % self.refit == 0:
                self.refit_pos()
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

        G_c = np.zeros((self.L, m, 1))
        G_c[:,-1,0] = self.dots[:,0]
        G_c[:,:-1,0] = np.matmul(self.geno[:,:].T, self.geno[:,self.pos[:m-1,0]])

        T_c = np.zeros((self.num_env, self.L, 1, 1))
        T_c[:,:,0,:] = np.matmul(self.geno[:,:].T, self.pheno)

        R_c,d_c = qr.make_rd_compact(G_c.copy(), T_c.copy(), self.R, self.d)

        S = self.S0 - (np.sum(self.d**2, axis=1) + d_c[:,:,0,0]**2)
        S_tot = np.sum(S, axis = 0)
        S_tot[R_c[:,-1,-1] < self.corr_cutoff] = np.inf

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
        print("p: {}".format(p))
        self.T = np.append(self.T, T_c[:,p,:,:], axis=1)
        self.d = np.append(self.d, d_c[:,p,:,:], axis=1)
        G_new = np.zeros((1,self.step+1,self.step+1))
        G_new[0,:-1,:-1] = self.G[0,:,:]
        G_new[0,:,-1,None] = G_c[p,:,:]
        self.G = G_new.copy()
        R_new = np.zeros((1,self.step+1,self.step+1))
        R_new[0,:-1,:-1] = self.R[0,:,:]
        R_new[0,:,-1,None] = R_c[p,:,:]
        self.R = R_new.copy()
        del G_c, T_c, R_c, d_c, G_new, R_new

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
        if self.epistatic:
            x = (self.pos[len(self.add_pos):] - len(self.add_pos))[:self.step+1,0]
            print(self.add_pos)
            print(x)
            print(self.e_loci[x])
            loci = np.hstack([self.add_pos, self.e_loci[x]])
            print(loci)
            df = pd.DataFrame(self.s_vals, index=x, columns=self.names)
            df.to_csv(outfile, sep="\t")
        else:
            df = pd.DataFrame(self.s_vals, index=self.pos[:self.step+1,0], columns=self.names)
            df.to_csv(outfile, sep="\t")

    def generate_s_vals(self):
        self.s_vals = np.zeros((self.step+1, self.num_env))
        R_inv = linalg.inv(self.R)
        self.s_vals[:,:] = np.matmul(R_inv,self.d).transpose((1,0,2))[:,:,0]

    def load_model(self, pos):
        self.pos = np.array(pos).astype("int")
        self.step = len(pos) - 1
        m = self.step + 1
        self.G = np.zeros([1,m,m])
        self.G = np.triu(np.matmul(self.geno[:,pos].T, self.geno[:,pos]))[None,:,:]
        self.T = np.zeros([self.num_env,m])
        self.T = np.sum(self.geno[None,:,pos] * self.pheno, axis=1)[:,:,None]
        self.T = np.matmul(self.pheno[:,:,0], self.geno[:,pos])[:,:,None]
        self.R, self.d = qr.make_rd(self.G.copy(), self.T.copy())

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

