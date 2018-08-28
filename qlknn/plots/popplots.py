import os
import re

import numpy as np
import pandas as pd
from IPython import embed
import matplotlib as mpl
#mpl.use('pdf')
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt
import seaborn as sns

from qlknn.NNDB.model import Network, NetworkJSON, PostprocessSlice
from qlknn.models.ffnn import QuaLiKizNDNN
from qlknn.dataset.data_io import sep_prefix
from qlknn.misc.analyse_names import is_transport, is_full_transport, is_pure, is_partial_heat, is_partial_particle, is_partial_rot, is_partial_diffusion, is_partial_momentum, is_leading
qlknn_root = os.path.abspath('../..')

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
        df = store[sep_prefix + varname]
        df.name = varname
        df.dropna(inplace=True)
        fig = plt.figure()
        ax = sns.distplot(df.loc[df.abs() < bound],
                          hist_kws={'log': False}, kde=True)

        sns.despine(ax=ax)
    return fig

def plot_dataset_dist(store, varname, cutoff=0.01, plot_zoomin=False):
    with sns.axes_style("white"):
        df = store[sep_prefix + varname]
        df.name = varname
        df.dropna(inplace=True)
        fig = plt.figure()
        ax = sns.distplot(df.loc[(df.quantile(cutoff) < df) &
                                 (df < df.quantile(1 - cutoff))],
                          hist_kws={'log': False}, kde=True)
        ax.set_ylabel('density')

        sns.despine(ax=ax)
        if plot_zoomin:
            loc = determine_subax_loc(ax)
            subax = inset_axes(ax,
                               width="30%",
                               height="30%",
                               loc=loc)
            if is_partial_particle(df.name):
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

def generate_store_name(set='training', unstable=True, gen=3, filter_id=8, dim=7):
    store_name = '{!s}_gen{!s}_{!s}D_nions0_flat_filter{!s}.h5.1'.format(set, gen, dim, filter_id)
    if unstable and set == 'training':
        store_name = '_'.join(['unstable', store_name])
    return store_name

def plot_pure_network_dataset_dist(self):
    filter_id = self.filter_id
    dim = len(net.feature_names)
    #store_name = 'unstable_training_gen{!s}_{!s}D_nions0_flat_filter{!s}.h5'.format(2, filter_id, dim)
    store_name = generate_store_name(True, 3, filter_id, dim)
    store = pd.HDFStore(os.path.join(qlknn_root, store_name))
    for train_dim in net.target_names:
        plot_dataset_dist(store, train_dim)
        deconstruct = re.split('_div_|_plus_', train_dim)
        if len(deconstruct) > 1:
            for sub_dim in deconstruct:
                plot_dataset_dist(store, sub_dim)
Network.plot_dataset_dist = plot_pure_network_dataset_dist

def generate_dataset_report(store, plot_pure=True, plot_heat=True, plot_particle=True, plot_rot=False, plot_full=False, plot_diffusion=False, plot_nonleading=False, plot_momentum=False, verbose_debug=False, plot_large_zoomin=False):
    with PdfPages('multipage_pdf.pdf') as pdf:
        for varname in store:
            varname = varname.replace(sep_prefix, '', 1)
            if verbose_debug:
                print(varname)
                print('is_full_transport', is_full_transport(varname))
                print('is_pure', is_pure(varname))
                print('is_partial_rot', is_partial_rot(varname))
                print('is_partial_heat', is_partial_heat(varname))
                print('is_partial_particle', is_partial_particle(varname))
                print('is_partial_diffusion', is_partial_diffusion(varname))
                print('is_leading', is_leading(varname))

            if (is_transport(varname) and
                ((plot_full and is_full_transport(varname)) or not is_full_transport(varname)) and
                ((plot_pure and is_pure(varname)) or not is_pure(varname)) and
                ((plot_rot and is_partial_rot(varname)) or not is_partial_rot(varname)) and
                ((plot_heat and is_partial_heat(varname)) or not is_partial_heat(varname)) and
                ((plot_particle and is_partial_particle(varname)) or not is_partial_particle(varname)) and
                ((plot_diffusion and is_partial_diffusion(varname)) or not is_partial_diffusion(varname)) and
                ((plot_momentum and is_partial_momentum(varname)) or not is_partial_momentum(varname)) and
                ((plot_nonleading and not is_leading(varname)) or is_leading(varname))):
                  #(not is_full_transport(varname) and
                  # not is_partial_rot(varname) and
                  # not is_partial_heat(varname) and
                  # not is_partial_particle(varname) and
                  # not is_partial_diffusion(varname) and
                  # is_leading(varname))):
                print('Plotting', varname)
                fig = plot_dataset_dist(store, varname)
                pdf.savefig(fig)
                plt.close(fig)
                if plot_large_zoomin:
                    try:
                        fig = plot_dataset_zoomin(store, varname)
                    except ZeroDivisionError:
                        fig = plt.figure()
                    pdf.savefig(fig)
                    plt.close(fig)

#net = Network.get_by_id(1409)

if __name__ == '__main__':
    store = pd.HDFStore(os.path.join(qlknn_root, generate_store_name(set='sane', unstable=False, dim=7)), 'r')
    generate_dataset_report(store)
    embed()
