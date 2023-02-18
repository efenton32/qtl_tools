import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import sys

from qtl_model import QTL_model

### Environments are read as command line inputs, eg:
### python name env1 env2 env3 env4

start = time.time()

geno_file = "PATH/TO/genofile.npy"
pheno_file = "PATH/TO/phenofile.csv"
save_path = "PATH/TO/SAVEPATH/"
pheno_df = pd.read_csv(phenofile, index_col = "barcode")

plot_name = sys.argv[1]
envs = sys.argv[2:]

print(plot_name)

pheno = pheno_df[envs].to_numpy()[:,:].T
geno = np.load(geno_file)[:,:]

for i in range(len(envs)):
    pheno[i,np.isnan(pheno[i,:])] = np.nanmin(pheno[i,:])

model = QTL_model(pheno, geno, names = envs)
model.run(num_qtl = 80, b_plots=True, path = savepath + plot_name + "/",
          b_save_itm = True, outfile = "qtl_results/{}.csv".format(plot_name))

model.dump_qtls("qtl_results/{}.csv".format(plot_name))

