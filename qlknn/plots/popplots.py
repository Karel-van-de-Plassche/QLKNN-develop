import pickle
import os
from itertools import product, chain, zip_longest
from collections import OrderedDict
from functools import partial
import re

from IPython import embed
import numpy as np
import pandas as pd
from peewee import AsIs, fn
import matplotlib as mpl
#mpl.use('pdf')
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt
import seaborn as sns

from qlknn.NNDB.model import Network, NetworkJSON, PostprocessSlice
from qlknn.models.ffnn import QuaLiKizNDNN, QuaLiKizDuoNN

def determine_subax_loc(ax, height_perc=.35, width_perc=.35):
    cover_left = False
    cover_right = False
    xlim = ax.get_xlim()
    full = np.sum(np.abs(xlim))
    left_bound = xlim[0] + width_perc * full
    right_bound = xlim[1] - width_perc * full
    top_bound = (1 - height_perc) * ax.get_ylim()[1]
    for child in ax.get_children():
        if isinstance(child, mpl.patches.Rectangle):
            xx = child.get_x()
            too_high = child.get_height() > 0.5 * ax.get_ylim()[1]
            if child.get_height() > top_bound:
                if xx < left_bound:
                    cover_left = True
                elif xx > right_bound:
                    cover_right = True

    if not cover_right:
        loc = 1
    elif not cover_left:
        loc = 2
    else:
        loc = 9
    return loc

def plot_dataset_zoomin(store, varname, bound=0.1):
    with sns.axes_style("white"):
        df = store[varname]
        df.dropna(inplace=True)
        fig = plt.figure()
        ax = sns.distplot(df.loc[df.abs() < bound],
                          hist_kws={'log': False}, kde=True)

        sns.despine(ax=ax)
    return fig

def plot_dataset_dist(store, varname, cutoff=0.01):
    with sns.axes_style("white"):
        df = store[varname]
        df.dropna(inplace=True)
        fig = plt.figure()
        ax = sns.distplot(df.loc[(df.quantile(cutoff) < df) &
                                 (df < df.quantile(1 - cutoff))],
                          hist_kws={'log': False}, kde=True)

        sns.despine(ax=ax)
        loc = determine_subax_loc(ax)
        subax = inset_axes(ax,
                           width="30%",
                           height="30%",
                           loc=loc)
        if 'pf' in df.name:
            quant_bound = .15
            low_bound = max(-1, df.quantile(quant_bound))
            high_bound = min(1, df.quantile(1 - quant_bound))
        else:
            low_bound = -1
            high_bound = 1
        sns.distplot(df.loc[(low_bound < df) & (df < high_bound)],
                     kde=True, kde_kws={'gridsize': 200},
                     ax=subax)
        if loc == 2:
            subax.yaxis.set_label_position("right")
            subax.yaxis.tick_right()
            sns.despine(ax=subax, left=True, right=False)
        else:
            sns.despine(ax=subax)
        subax.set_xlabel('')
    return fig

def generate_store_name(unstable=True, gen=2, filter_id=7, dim=7):
    store_name = 'training_gen{!s}_{!s}D_nions0_flat_filter{!s}.h5'.format(gen, dim, filter_id)
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
        deconstruct = re.split('_div_|_plus_', train_dim)
        if len(deconstruct) > 1:
            for sub_dim in deconstruct:
                plot_dataset_dist(store, sub_dim)
Network.plot_dataset_dist = plot_pure_network_dataset_dist

def generate_dataset_report(store, plot_rot=False, plot_full=False, plot_diffusion=False, plot_nonleading=False):
    is_full = lambda name: all(sub not in name for sub in ['TEM', 'ITG', 'ETG'])
    is_rot = lambda name: any(sub in name for sub in ['vr', 'vf'])
    is_diffusion = lambda name: any(sub in name for sub in ['df', 'vc', 'vt'])
    def is_leading(name):
        leading = True
        if not is_full(name):
            if any(sub in name for sub in ['div', 'plus']):
                if 'ITG' in name:
                    if name not in ['efeITG_GB_div_efiITG_GB', 'pfeITG_GB_div_efiITG_GB']:
                        leading = False
                elif 'TEM' in name:
                    if name not in ['efiTEM_GB_div_efeTEM_GB', 'pfeTEM_GB_div_efeTEM_GB']:
                        leading = False
            if 'pfi' in name:
                leading = False
        return leading

    with PdfPages('multipage_pdf.pdf') as pdf:
        for varname in store:
            varname = varname.lstrip('/')
            if ((varname not in ['input', 'constants']) and
                ((plot_rot and is_rot(varname)) or
                 (plot_full and is_full(varname)) or
                 (plot_diffusion and is_diffusion(varname)) or
                 (plot_nonleading and not is_leading(varname)) or
                 (not is_rot(varname) and not is_full(varname) and not is_diffusion(varname) and is_leading(varname)))):
                print(varname)
                fig = plot_dataset_dist(store, varname)
                pdf.savefig(fig)
                plt.close(fig)
                fig = plot_dataset_zoomin(store, varname)
                pdf.savefig(fig)
                plt.close(fig)

#net = Network.get_by_id(1409)

if __name__ == '__main__':
    store = pd.HDFStore(os.path.join(qlknn_root, generate_store_name(dim=7)))
    generate_dataset_report(store)
    embed()
