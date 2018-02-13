from IPython import embed
import numpy as np
import pandas as pd
from itertools import product, chain, zip_longest
import pickle
import os
import sys
import time
qlknn_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
networks_path = os.path.join(qlknn_root, 'networks')
NNDB_path = os.path.join(qlknn_root, 'NNDB')
training_path = os.path.join(qlknn_root, 'training')
plot_path = os.path.join(qlknn_root, 'plots')
sys.path.append(networks_path)
sys.path.append(NNDB_path)
sys.path.append(training_path)
from model import Network, NetworkJSON, PostprocessSlice, ComboNetwork, MultiNetwork
from run_model import QuaLiKizNDNN, QuaLiKizDuoNN
from train_NDNN import shuffle_panda, normab, normsm
from functools import partial

import matplotlib as mpl
mpl.use('pdf')
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid.inset_locator import inset_axes
import matplotlib.pyplot as plt
from load_data import load_data, load_nn, prettify_df
from collections import OrderedDict
from peewee import AsIs, fn
import re

import seaborn as sns

def plot_dataset_dist(store, varname, cutoff=0.01):
    with sns.axes_style("white"):
        start = time.time()
        df = store[varname]
        df.dropna(inplace=True)
        fig = plt.figure()
        ax = sns.distplot(df, hist_kws={'range': df.quantile([.01, 1 - cutoff]), 'log': False}, kde=False)
        sns.despine()
        subax = inset_axes(ax,
                           width="30%",
                           height="30%")
        sns.distplot(df,  hist_kws={'range': [-1, 1]}, kde=False, ax=subax)
    return fig

def generate_store_name(unstable=True, gen=2, filter_id=7, dim=7):
    store_name = 'training_gen{!s}_{!s}D_nions0_flat_filter{!s}.h5'.format(gen, filter_id, dim)
    if unstable:
        store_name = '_'.join(['unstable', store_name])
    return store_name

def plot_pure_network_dataset_dist(self):
    filter_id = self.filter_id
    dim = len(net.feature_names)
    #store_name = 'unstable_training_gen{!s}_{!s}D_nions0_flat_filter{!s}.h5'.format(2, filter_id, dim)
    store_name = generate_store_name(True, 2, filter_id, dim)
    store = pd.HDFStore(os.path.join(qlknn_root, store_name))
    embed()
    for train_dim in net.target_names:
        plot_dataset_dist(store, train_dim)
        deconstruct = re.split('_div_|_add_', train_dim)
        if len(deconstruct) > 1:
            for sub_dim in deconstruct:
                plot_dataset_dist(store, sub_dim)
Network.plot_dataset_dist = plot_pure_network_dataset_dist

def generate_dataset_report(store, plot_rot=False, plot_full=False):
    is_full = lambda name: all(sub not in name for sub in ['TEM', 'ITG', 'ETG'])
    is_rot = lambda name: any(sub in name for sub in ['vr', 'vf'])
    with PdfPages('multipage_pdf.pdf') as pdf:
        for varname in store:
            varname = varname.lstrip('/')
            if ((varname not in ['input', 'constants']) and
                ((plot_rot and is_rot(varname)) or
                 (plot_full and is_full(varname)) or
                 (not is_rot(varname) and not is_full(varname)))):
                print(varname)
                fig = plot_dataset_dist(store, varname)
                pdf.savefig(fig)
                plt.close(fig)

#net = Network.get_by_id(1409)

store = pd.HDFStore(os.path.join(qlknn_root, generate_store_name()))
generate_dataset_report(store)
embed()
