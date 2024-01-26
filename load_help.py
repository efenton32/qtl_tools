import numpy as np
import pandas as pd

def load_qtls(fname):
    """
    Loads a QTL model
    
    Paramters:
        fname [str]: Path to a saved QTL file, in csv format
    """
    return pd.read_csv(fname, delimiter="\t", index_col=0)

def load_qtls(f_name):
    return pd.read_csv(f_name, delimiter="\t", index_col=0)

def to_array(idx_str):
    return np.array( idx_str.strip("[] ").split())

def get_envs(hub, grad, ret_vals = False):
    all_envs = []
    with open("/n/holystore01/LABS/desai_lab/Everyone/eliot/sepsis/sepsis_data/env_list_reordered_filtered.csv") as f:
        for line in f:
            all_envs.append(line.strip())
    envs = []
    vals = []
    if grad == "kcl" and hub == "ypd":
        envs = ["ypd_temp_30", "ypd_kcl_02", "ypd_kcl_04", "ypd_kcl_08", "ypd_kcl_1", "ypd_kcl_15"]
        vals = [0, 2, 4, 8, 10, 15]
    elif grad == "ph" and hub == "ph":
        envs = ["ypd_ph_263", "ypd_ph_4", "ypd_ph_55", "ypd_ph_7"]
        vals = [2.63, 4.0, 5.5, 7.0]
    elif grad != "temp":
        envs.append(hub+"_temp_30")
        vals.append(0)
        for env in all_envs:
            if hub in env and grad in env:
                envs.append(env)
                vals.append(int(env.split("_")[2]))
    else:
        for env in all_envs:
            if hub in env and grad in env:
                envs.append(env)
                vals.append(int(env.split("_")[2]))
    if ret_vals:
        return envs, vals
    else:
        return envs

"""
def reindex(df):
    new_index = []
    for idx in df.index:
        if idx[0] == "[":
            new_index.append(to_array(idx))
        else:
            new_index.append(np.array([idx, idx]).astype("int"))
    return df.reindex(new_index)
"""

def get_additive(idx, df):
    if type(idx) == str:
        idx = to_array(idx)
    return df.loc[idx]

def get_epistatic(idx, df):
    idx = str(idx)
    locs = []
    for i in df.index:
        if "[" in i and idx in i:
            locs.append(i)
    return locs

