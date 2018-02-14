from IPython import embed
import numpy as np
import scipy.stats as stats
import pandas as pd

import os
import sys
networks_path = os.path.abspath(os.path.join((os.path.abspath(__file__)), '../../networks'))
NNDB_path = os.path.abspath(os.path.join((os.path.abspath(__file__)), '../../NNDB'))
training_path = os.path.abspath(os.path.join((os.path.abspath(__file__)), '../../training'))
sys.path.append(networks_path)
sys.path.append(NNDB_path)
sys.path.append(training_path)
from model import Network, NetworkJSON, Hyperparameters
from run_model import QuaLiKizNDNN
from train_NDNN import shuffle_panda

from peewee import Param, Passthrough

import matplotlib.pyplot as plt
from matplotlib import gridspec
from load_data import load_data, load_nn

def results_to_sorted(varname, df):
    df = pd.DataFrame(df, columns=['id', varname, 'rms_test'])
    if df[varname].__class__ == list:
        df[varname] = df[varname].apply(tuple)
    df.sort_values([varname, 'rms_test'], inplace = True)
    df_trim = pd.DataFrame(columns=['id', varname, 'rms_test'])
    for index, row in df.iterrows():
        df_best = df.iloc[df.loc[(df[varname] == row[varname])].index[0]]
        df_best = df.loc[df.loc[(df[varname] == row[varname])].index[0]]
        if df[varname].__class__ == tuple:
            is_not_in =  ~(df_best[varname] == df_trim[varname]).any()
        else:
            is_not_in =  ~(df_trim[varname].isin(df_best)).any()
        if is_not_in:
            df_trim = df_trim.append(df_best)
    return df_trim
