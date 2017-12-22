from IPython import embed
from multiprocessing import Pool, cpu_count
#import mega_nn
import numpy as np
import scipy.stats as stats
import pandas as pd
from itertools import product, chain, zip_longest
import pickle
import os
import sys
import time
networks_path = os.path.abspath(os.path.join((os.path.abspath(__file__)), '../../networks'))
NNDB_path = os.path.abspath(os.path.join((os.path.abspath(__file__)), '../../NNDB'))
training_path = os.path.abspath(os.path.join((os.path.abspath(__file__)), '../../training'))
sys.path.append(networks_path)
sys.path.append(NNDB_path)
sys.path.append(training_path)
from model import Network, NetworkJSON, PostprocessSlice, ComboNetwork, MultiNetwork, no_elements_in_list, elements_in_list
from run_model import QuaLiKizNDNN, QuaLiKizDuoNN
from train_NDNN import shuffle_panda
from functools import partial

import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
pretty = False
from load_data import nameconvert
if pretty:
    plt.style.use('./thesis.mplstyle')
    mpl.rcParams.update({'font.size': 16})
else:
    nameconvert = {name: name for name in nameconvert}

from matplotlib import gridspec, cycler
from load_data import load_data, load_nn, prettify_df
from collections import OrderedDict
from peewee import Param, fn
import re

def mode_to_settings(mode):
    settings = {}
    if mode == 'debug':
        settings['plot']             = True
        settings['plot_pop']         = True
        settings['plot_nns']         = True
        settings['plot_slice']       = True
        settings['plot_poplines']    = True
        settings['plot_threshlines'] = True
        settings['plot_zerocolors']  = False
        settings['plot_thresh1line'] = False
        settings['calc_thresh1']     = False
        settings['hide_qualikiz']    = False
        settings['debug']            = True
        settings['parallel']         = False
    elif mode == 'quick':
        settings['plot']             = False
        settings['plot_pop']         = False
        settings['plot_nns']         = False
        settings['plot_slice']       = False
        settings['plot_poplines']    = False
        settings['plot_threshlines'] = False
        settings['plot_zerocolors']  = False
        settings['plot_thresh1line'] = False
        settings['calc_thresh1']     = False
        settings['hide_qualikiz']    = False
        settings['debug']            = False
        settings['parallel']         = True
    elif mode == 'pretty':
        settings['plot']             = True
        settings['plot_pop']         = False
        settings['plot_nns']         = True
        settings['plot_slice']       = False
        settings['plot_poplines']    = False
        settings['plot_threshlines'] = False
        settings['plot_zerocolors']  = False
        settings['plot_thresh1line'] = False
        settings['calc_thresh1']     = False
        settings['hide_qualikiz']    = False
        settings['debug']            = True
        settings['parallel']         = False
    return settings

def get_similar_not_in_table(table, max=20, only_sep=False, no_particle=False):
    for cls, field_name in [(Network, 'network'),
                (ComboNetwork, 'combo_network'),
                (MultiNetwork, 'multi_network')
                ]:
        tags = ["div", "plus"]
        if no_particle is True:
            tags.append('pf')
        non_sliced = no_elements_in_list(cls, tags)
        if only_sep is True:
            non_sliced &= elements_in_list(cls, ['TEM', 'ITG', 'ETG'])
        non_sliced &= (cls
                      .select()
                      .where(~fn.EXISTS(table.select().where(getattr(table, field_name) == cls.id)))
                     )
        if non_sliced.count() > 0:
            network = non_sliced.get()
            break

    non_sliced &= (cls.select()
                  .where(cls.target_names == Param(network.target_names))
                  .where(cls.feature_names == Param(network.feature_names))
                  )
    non_sliced = non_sliced.limit(max)
    return non_sliced

def nns_from_NNDB(max=20):
    non_sliced = get_similar_not_in_table(PostprocessSlice, max=max, only_sep=True, no_particle=True)
    network = non_sliced.get()
    style = 'mono'
    if len(network.target_names) == 2:
        match_0 = re.compile('^(.f)(.)(ITG|ETG|TEM)_GB').findall(network.target_names[0])
        match_1 = re.compile('^(.f)(.)(ITG|ETG|TEM)_GB').findall(network.target_names[1])
        if len(match_0) == 1 and len(match_1) == 1:
            group_0 = match_0[0]
            group_1 = match_1[0]
            if ((group_0[1] == 'e' and group_1[1] == 'i') or
                (group_0[1] == 'i' and group_1[1] == 'e')):
                style='duo'
            else:
                raise Exception('non-matching target_names. Not sure what to do.. {s}'
                                .format(network.target_names))
    matches = []
    for target_name in network.target_names:
        matches.extend(re.compile('^.f.(ITG|ETG|TEM)_GB').findall(target_name))
    if matches[1:] == matches[:-1]:
        if matches[0] == 'ITG':
            slicedim = 'Ati'
        elif matches[0] == 'TEM' or matches[0] == 'ETG':
            slicedim = 'Ate'
    else:
        raise Exception('Unequal stability regime. Cannot determine slicedim')
    nn_list = {network.id: str(network.id) for network in non_sliced}
    print('Found {:d} {!s} with target {!s}'.format(non_sliced.count(), network.__class__, network.target_names))

    nns = OrderedDict()
    for dbnn in non_sliced:
        nn = dbnn.to_QuaLiKizNN()
        nn.label = '_'.join([str(el) for el in [dbnn.__class__.__name__ , dbnn.id]])
        nns[nn.label] = nn
    return slicedim, style, nns


def populate_nn_list(nn_set):
    if nn_set == 'c_L2':
        nn_list = OrderedDict([(61, '$c_{L2} = 0.0$'),
        #                       (48, '$c_{L2} = 0.05$'),
                               (37, '$c_{L2} = 0.1$'),
        #                       (50, '$c_{L2} = 0.2$'),
        #                       (51, '$c_{L2} = 0.35$'),
                               (49, '$c_{L2} = 0.5$'),
        #                       (52, '$c_{L2} = 1.0$'),
                               (53, '$c_{L2} = 2.0$')])
        slicedim = 'Ate'
        style = 'mono'
    elif nn_set == 'topo':
        nn_list = OrderedDict([(65, 'neurons = $(10, 10)$'),
                               (64, 'neurons = $(30, 30)$'),
                               (73, 'neurons = $(30, 30, 30)$'),
                               (83, 'neurons = $(45, 45)$'),
                               (34, 'neurons = $(60, 60)$'),
                               (38, 'neurons = $(80, 80)$'),
                               (66, 'neurons = $(120, 120)$')])
        slicedim = 'Ate'
        style = 'mono'
    elif nn_set == 'filter':
        #nn_list = OrderedDict([(37, 'filter = 3'),
        #                       (58, 'filter = 4'),
        #                       (60, 'filter = 5')])
        nn_list = OrderedDict([(37, '$max(\chi_{ETG,e}) = 60$'),
                               (60, '$max(\chi_{ETG,e}) = 100$')])
        slicedim = 'Ate'
        style = 'mono'
    elif nn_set == 'goodness':
        nn_list = OrderedDict([(62, 'goodness = mabse'),
                               (37, 'goodness = mse')])
        slicedim = 'Ate'
        style = 'mono'
    elif nn_set == 'early_stop':
        nn_list = OrderedDict([(37, 'stop measure = loss'),
                               #(11, '$early_stop = mse'),
                               (18, 'stop measure = MSE')])
        slicedim = 'Ate'
        style = 'mono'
    elif nn_set == 'similar':
        nn_list = OrderedDict([
                               (37, '37'),
                               (67, '67'),
                               (68, '68'),
                               (69, '69'),
                               (70, '70'),
                               (71, '71'),
                               (72, '72'),
                               (73, '73'),
                               (74, '74'),
                               ])
        slicedim = 'Ate'
        style = 'mono'
    elif nn_set == 'best':
        nn_list = OrderedDict([(46, '')]) #efeETG
        nn_list = OrderedDict([(88, '')]) #efiITG
        slicedim = 'Ate'
        style = 'mono'

    elif nn_set == 'duo':
        nn_list = OrderedDict([
            (205, 'es_20'),
            (204, 'es_5'),
            (203, 'es_wrong')
            ])
        slicedim = 'Ati'
        style = 'duo'

    return slicedim, style, nn_list

def nns_from_nn_list(nn_list, slicedim, labels=True):
    nns = OrderedDict()
    for nn_index, nn_label in nn_list.items():
        nn = nns[nn_index] = load_nn(nn_index)
        if labels:
            nn.label = nn_label
        else:
            nn.label = ''
    return nns

def nns_from_manual():
    nns = OrderedDict()

    #div_nn = load_nn(405)
    #sum_nn = load_nn(406)
    #nn = QuaLiKizDuoNN(['efiITG_GB', 'efeITG_GB'], div_nn, sum_nn, [lambda x, y: x * y/(x + 1), lambda x, y: y/(x + 1)])
    #nn.label = 'div_style'
    #nns[nn.label] = nn

    #nn_efi = load_nn(88)
    #nn_efe = load_nn(89)
    #nn = QuaLiKizDuoNN(['efiITG_GB', 'efeITG_GB'], nn_efi, nn_efe, [lambda x, y: x, lambda x, y: y])
    #nn.label = 'sep_style'
    #nns[nn.label] = nn

    #nn = load_nn(205)
    #nn.label = 'combo_style'
    #nns[nn.label] = nn

    #subnn = (ComboNetwork.select()
    #            .where(ComboNetwork.id == 78)
    #            ).get()
    #nn = subnn.to_QuaLiKizComboNN()
    #nn.label = 'bla'
    #nns[nn.label] = nn

    #dbnn = Network.by_id(135).get()

    dbnns = []
    #dbnns.append(MultiNetwork.by_id(119).get())
    dbnns.append(Network.by_id(549).get())
    #dbnns.append(ComboNetwork.by_id(1050).get())
    #dbnns.append(MultiNetwork.by_id(102).get())

    for dbnn in dbnns:
        nn = dbnn.to_QuaLiKizNN()
        nn.label = '_'.join([str(el) for el in [dbnn.__class__.__name__ , dbnn.id]])
        nns[nn.label] = nn

    slicedim = 'Ati'
    style='duo'
    style='mono'
    return slicedim, style, nns


def prep_df(input, data, nns, filter_less=np.inf, filter_geq=-np.inf, shuffle=True):
    nn0 = list(nns.values())[0]
    target_names = nn0._target_names
    feature_names = nn0._feature_names
    input = input[feature_names]
    if 'gam_less_GB' in data:
        df = pd.DataFrame({'leq': data['gam_leq_GB'],
                           'less': data['gam_less_GB']})
    else:
        df = pd.DataFrame({'leq': data['gam_leq_GB'],
                           'great': data['gam_great_GB']})

    df = (df.max(axis=1)
          .to_frame('maxgam')
    )
    df = input.join([data[target_names], df['maxgam']])

    #itor = zip(['An', 'Ate', 'Ti_Te', 'qx', 'smag', 'x'], ['0.00', '10.00', '1.00', '5.00', '0.40', '0.45'])
    #for name, val in itor:
    #    df = df[np.isclose(df[name], float(val),     atol=1e-5, rtol=1e-3)]

    df = df[(df[target_names] < filter_less).all(axis=1)]
    df = df[(df[target_names] >= filter_geq).all(axis=1)]
    #print(np.sum(df['target'] < 0)/len(df), ' frac < 0')
    #print(np.sum(df['target'] == 0)/len(df), ' frac == 0')
    #print(np.sum(df['target'] > 0)/len(df), ' frac > 0')
    #uni = {col: input[col].unique() for col in input}
    #uni_len = {key: len(value) for key, value in uni.items()}
    #input['index'] = input.index
    df.set_index([col for col in input], inplace=True)
    df = df.astype('float64')
    df = df.sort_index(level=slicedim)
    df = df.unstack(slicedim)
    if shuffle:
        df = shuffle_panda(df)
    #df.sort_values('smag', inplace=True)
    #input, data = prettify_df(input, data)
    #input = input.astype('float64')
    # Filter

    #df = df.iloc[1040:2040,:]
    print('dataset loaded!')
    return df, target_names

def is_unsafe(df, nns):
    unsafe = True
    for nn in nns.values():
        slicedim_idx = nn._feature_names[nn._feature_names == slicedim].index[0]
        varlist = list(df.index.names)
        varlist.insert(slicedim_idx, slicedim)
        try:
            if ~np.all(varlist == nn._feature_names):
                unsafe = False
        except ValueError:
            raise Exception('Dataset has features {!s} but dataset has features {!s}'.format(varlist, list(nn._feature_names)))
    return unsafe


def calculate_thresh1(x, feature, target, debug=False):
    try:
        idx = target.index[target == 0][-1] #index of last zero
        slope, intercept, r_value, p_value, std_err = stats.linregress(feature[(target.index > idx) & ~target.isnull()], target[(target.index > idx) & ~target.isnull()])
        thresh_pred = x * slope + intercept
        thresh1 = x[thresh_pred < 0][-1]
    except (ValueError, IndexError):
        thresh1 = np.NaN
        if debug:
            print('No threshold1')
    return thresh1

def calculate_thresh2(feature, target, debug=False):
    if len(target.shape) > 1:
        raise NotImplementedError('2D threshold not implemented yet')
    try:
        idx = np.where(target == 0)[0][-1] #Only works for 1D
        #idx = np.arange(target.shape[0]),target.shape[1] - 1 - (target[:,::-1]==0).argmax(1) #Works for 2D
        thresh2 = np.mean(feature[idx:idx+2])
    except IndexError:
        thresh2 = np.NaN
        if debug:
            print('No threshold2')

    return thresh2
#5.4 ms ± 115 µs per loop (mean ± std. dev. of 7 runs, 100 loops each) total
def process_chunk(target_names, chunck, settings=None, unsafe=False):

    res = []
    for ii, row in enumerate(chunck.iterrows()):
        res.append(process_row(target_names, row, settings=settings, unsafe=unsafe))
    return res

def process_row(target_names, row, ax1=None, unsafe=False, settings=None):
    index, slice_ = row
    feature = slice_.index.levels[1]
    #target = slice.loc[target_names]
    target = slice_.values[:len(feature) * len(target_names)].reshape(len(target_names), len(feature))
    if np.all(target == 0):
        return (1,)
    else:
        # 156 µs ± 10.4 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each) (no zerocolors)
        thresh_nn = np.empty(len(target_names) * len(nns))
        thresh_nn_i = np.empty_like(thresh_nn, dtype='int64')
        popbacks = np.empty_like(thresh_nn)
        thresh1_misses = np.empty_like(thresh_nn)
        thresh2_misses = np.empty_like(thresh_nn)
        if settings['plot_zerocolors']:
            maxgam = slice_['maxgam']

        # Create slice, assume sorted
        # 14.8 µs ± 1.27 µs per loop (mean ± std. dev. of 7 runs, 100000 loops each)
        x = np.linspace(feature.values[0],
                        feature.values[-1],
                        200)
        #if plot:
        if not ax1 and settings['plot']:
            fig = plt.figure()
            if settings['plot_pop'] and settings['plot_slice']:
                gs = gridspec.GridSpec(2, 2, height_ratios=[10, 1], width_ratios=[5,1],
                                    left=0.05, right=0.95, wspace=0.05, hspace=0.05)
                ax2 = plt.subplot(gs[1,0])
                ax3 = plt.subplot(gs[0,1])
            if not settings['plot_pop'] and settings['plot_slice']:
                gs = gridspec.GridSpec(2, 1, height_ratios=[10, 2], width_ratios=[1],
                                    left=0.05, right=0.95, wspace=0.05, hspace=0.05)
                ax2 = plt.subplot(gs[1,0])
            if not settings['plot_pop'] and not settings['plot_slice']:
                gs = gridspec.GridSpec(1, 1, height_ratios=[1], width_ratios=[1],
                                    left=0.05, right=0.95, wspace=0.05, hspace=0.05)
            ax1 = plt.subplot(gs[0,0])
            #ax1.set_prop_cycle(cycler('color', ['#f1eef6','#d7b5d8','#df65b0','#dd1c77','#980043']))
            # http://tristen.ca/hcl-picker/#/clh/5/273/2A0A75/D59FEB
            #ax1.set_prop_cycle(cycler('color', ['#2A0A75','#6330B8','#9F63E2','#D59FEB']))
            if len(nns) == 1:
                color_range = np.array([.7])
            else:
                color_range = np.linspace(0, 0.9, len(nns))
            ax1.set_prop_cycle(cycler('color', plt.cm.plasma(color_range)))
            ax1.set_xlabel(nameconvert[slicedim])
            ax1.set_ylabel(nameconvert[list(nns.items())[0][1]._target_names[0]])
        if settings['calc_thresh1']:
            thresh1 = calculate_thresh1(x, feature, target, debug=settings['debug'])
            print('whyyy?')

        # 12.5 µs ± 970 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
        thresh2 = calculate_thresh2(feature.values, target[0,:], debug=settings['debug'])

        if settings['plot'] and settings['plot_threshlines']:
            ax1.axvline(thresh2, c='black', linestyle='dashed')

        # 13.7 µs ± 1.1 µs per loop (mean ± std. dev. of 7 runs, 100000 loops each)
        if unsafe:
            slice_list = [np.full_like(x, val) for val in index]
            slicedim_idx = np.nonzero(list(nns.values())[0]._feature_names.values == slicedim)[0][0]
            slice_list.insert(slicedim_idx, x)
        else:
            slice_dict = {name: np.full_like(x, val) for name, val in zip(df.index.names, index)}
            slice_dict[slicedim] = x



        # Plot target points
        if settings['plot'] and settings['plot_slice']:
            table = ax2.table(cellText=[[nameconvert[name] for name in df.index.names],
                                        ['{:.2f}'.format(xx) for xx in index]],cellLoc='center')
            table.auto_set_font_size(False)
            table.scale(1, 1.5)
            #table.set_fontsize(20)
            ax2.axis('tight')
            ax2.axis('off')
        #fig.subplots_adjust(bottom=0.2, transform=ax1.transAxes)


        # Plot nn lines
        nn_preds = np.ndarray([x.shape[0], 0])
        for ii, (nn_index, nn) in enumerate(nns.items()):
            if all(['ef' in name for name in nn._target_names]):
                clip_low = True
                low_bound = np.zeros((len(nn._target_names), 1))

                #high_bound = np.full((len(nn._target_names), 1), np.inf)
                clip_high = False
                high_bound = None
            else:
                raise NotImplementedError('Particle bounds')
                clip_low = True
                low_bound = np.full((len(nn._target_names), 1), -80)
                clip_high = True
                high_bound = np.full((len(nn._target_names), 1), 80)
            if unsafe:
                nn_pred = nn.get_output(np.array(slice_list).T, clip_low=clip_low, low_bound=low_bound, clip_high=clip_high, high_bound=high_bound, safe=not unsafe, output_pandas=False)
            else:
                nn_pred = nn.get_output(pd.DataFrame(slice_dict), clip_low=clip_low, low_bound=low_bound, clip_high=clip_high, high_bound=high_bound, safe=not unsafe, output_pandas=True).values
            nn_preds = np.concatenate([nn_preds, nn_pred], axis=1)

        if settings['plot'] and settings['plot_nns']:
            lines = []
            if style == 'duo':
                labels = np.repeat([nn.label for nn in nns.values()], 2)
                for ii in range(0, nn_preds.shape[1], 2):
                    lines.append(ax1.plot(x, nn_preds[:, ii], label=labels[ii])[0])
                    lines.append(ax1.plot(x, nn_preds[:, ii+1], label=labels[ii+1], c=lines[-1].get_color(), linestyle='dashed')[0])
            else:
                for ii, (nn, row) in enumerate(zip(nns.values(), nn_preds.T)):
                    lines.append(ax1.plot(x, row, label=nn.label)[0])

        matrix_style = False
        if matrix_style:
            thresh_i = (np.arange(nn_preds.shape[1]),nn_preds.shape[0] - 1 - (nn_preds[::-1,:]==0).argmax(0))[1]
            thresh = x[thresh_i]
            thresh[thresh == x[-1]] = np.nan
        else:
            for ii, row in enumerate(nn_preds.T):
                try:
                    if row[-1] == 0:
                        thresh_nn[ii] = np.nan
                    else:
                        thresh_i = thresh_nn_i[ii] = np.where(row == 0)[0][-1]
                        thresh_nn[ii] = x[thresh_i]
                except IndexError:
                    thresh_nn[ii] = np.nan

        if settings['plot'] and settings['plot_threshlines']:
            for ii, row in enumerate(thresh_nn):
                ax1.axvline(row, c=lines[ii].get_color(), linestyle='dotted')
                if settings['debug']:
                    print('network ', ii, 'threshold ', row)



        if matrix_style:
            masked = np.ma.masked_where(x[:, np.newaxis] > thresh, nn_preds)
            #popback_i = (masked.shape[0] - 1 - (masked[::1,:]!=0)).argmax(0)
            popback_i = masked.shape[0] - 1 - (masked.shape[0] - 1 - (masked[::-1,:]!=0)).argmin(0)
            popback = x[popback_i]
            popback[popback == x[-1]] = np.nan
        else:
            for ii, row in enumerate(nn_preds.T):
                if not np.isnan(thresh_nn[ii]):
                    try:
                        popback_i = np.flatnonzero(row[:thresh_nn_i[ii]])
                        popbacks[ii] = x[popback_i[-1]]
                    except (IndexError):
                        popbacks[ii] = np.nan
                else:
                    popbacks[ii] = np.nan

        # 5.16 µs ± 188 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

        if settings['plot'] and settings['plot_pop']:
            thresh2_misses = thresh_nn - thresh2
            thresh2_popback = popbacks - thresh2
            slice_stats = np.array([thresh2_misses, thresh2_popback]).T
            slice_strings = np.array(['{:.1f}'.format(xx) for xx in slice_stats.reshape(slice_stats.size)])
            slice_strings = slice_strings.reshape(slice_stats.shape)
            slice_strings = np.insert(slice_strings, 0, ['thre_mis', 'pop_mis'], axis=0)
            table = ax3.table(cellText=slice_strings, loc='center')
            table.auto_set_font_size(False)
            ax3.axis('tight')
            ax3.axis('off')
            if settings['debug']:
                print(slice_stats.flatten())

        if settings['plot']:
            if settings['plot_zerocolors']:
                color = target.copy()
                color[(target == 0) & (maxgam == 0)] = 'green'
                color[(target != 0) & (maxgam == 0)] = 'red'
                color[(target == 0) & (maxgam != 0)] = 'magenta'
                color[(target != 0) & (maxgam != 0)] = 'blue'
            else:
                color='blue'
            if settings['hide_qualikiz']:
                color='white'
                zorder=1
                label=''
            else:
                zorder=1000
                label = 'Turbulence model'
                label=''
            markers = ['x', '+']
            for column, marker in zip(target, markers):
                ax1.scatter(feature[column != 0],
                            column[column != 0], c=color, label=label, marker=marker, zorder=zorder)
            ax1.scatter(feature[column==0],
                        column[column==0], edgecolors=color, label=label, marker='o', facecolors='none', zorder=zorder)

        # Plot regression
        if settings['plot'] and settings['plot_thresh1line'] and not np.isnan(thresh1):
            #plot_min = ax1.get_ylim()[0]
            plot_min = -0.1
            x_plot = x[(thresh_pred > plot_min) & (thresh_pred < ax1.get_ylim()[1])]
            y_plot = thresh_pred[(thresh_pred > plot_min) & (thresh_pred < ax1.get_ylim()[1])]
            ax1.plot(x_plot, y_plot, c='gray', linestyle='dotted')
            ax1.plot(x[x< thresh1], np.zeros_like(x[x< thresh1]), c='gray', linestyle='dotted')
            #ax1.axvline(thresh1, c='black', linestyle='dotted')

        slice_res = np.array([thresh_nn, popbacks]).T
        if settings['plot']:
            ax1.legend()
            ax1.set_ylim(bottom=min(ax1.get_ylim()[0], 0))
            plt.show()
            fig.savefig('slice.pdf', format='pdf', bbox_inches='tight')
            qlk_data = pd.DataFrame(target.T, columns=target_names, index=feature)
            nn_data = pd.DataFrame(nn_preds, columns=pd.MultiIndex.from_product([[nn.label for nn in nns.values()], target_names]))
            nn_data.index.name = feature.name
            embed()
            plt.close(fig)
        return (0, thresh2, slice_res.flatten())
    #sliced += 1
    #if sliced % 1000 == 0:
    #    print(sliced, 'took ', time.time() - starttime, ' seconds')
def extract_stats(totstats, style):
    df = totstats.copy()
    df = df.reorder_levels([2,0,1], axis=1)

    results = pd.DataFrame()

    for relabs, measure in zip(['rel', 'abs'], ['thresh', 'pop']):
        df2 = df[measure]
        qlk_data = df2['QLK']
        network_data = df2.drop('QLK', axis=1)
        if relabs == 'rel':
            mis = network_data.subtract(qlk_data, level=1).divide(qlk_data, level=1)
        elif relabs == 'abs':
            mis = network_data.subtract(qlk_data, level=1)

        quant1 = 0.025
        quant2 = 1 - quant1
        quant = mis.quantile([quant1, quant2])
        results['_'.join([measure, relabs, 'mis', 'median'])] = mis.median()
        results['_'.join([measure, relabs, 'mis', '95width'])] = quant.loc[quant2] - quant.loc[quant1]

        results['_'.join(['no', measure, 'frac'])] = mis.isnull().sum() / len(mis)

    if style == 'duo':
        duo_results = pd.DataFrame()
        measure = 'thresh'
        df2 = df[measure]
        network_data = df2.drop('QLK', axis=1)
        network_data = network_data.reorder_levels([1, 0], axis=1)
        efelike_name = network_data.columns[1][0]
        efilike_name = network_data.columns[0][0]
        mis = network_data[efilike_name] - network_data[efelike_name]
        quant = mis.quantile([quant1, quant2])
        duo_results['dual_thresh_mismatch_median'] = mis.median()
        duo_results['dual_thresh_mismatch_95width'] = quant.loc[quant2] - quant.loc[quant1]
        duo_results['no_dual_thresh_frac'] = mis.isnull().sum() / len(mis)
    else:
        duo_results = pd.DataFrame()
    return results, duo_results

def extract_nn_stats(results, duo_results, nns, submit_to_nndb=False):
    for network_name, res in results.unstack().iterrows():
        network_class, network_number = network_name.split('_')
        nn = nns[network_name]
        if network_class == 'Network':
            res_dict = {'network': network_number}
        elif network_class == 'ComboNetwork':
            res_dict = {'combo_network': network_number}
        elif network_class == 'MultiNetwork':
            res_dict = {'multi_network': network_number}
        for stat, val in res.unstack(level=0).iteritems():
            res_dict[stat] = val.loc[nn._target_names].values

        try :
            duo_res = duo_results.loc[network_name]
            res_dict.update(duo_res)
        except KeyError:
            pass

        postprocess_slice = PostprocessSlice(**res_dict)
        if submit_to_nndb is True:
            postprocess_slice.save()

if __name__ == '__main__':
    nn_set = 'duo'
    nn_set = 'best'
    mode = 'pretty'
    mode = 'debug'
    submit_to_nndb = False
    mode = 'quick'
    submit_to_nndb = True

    store = pd.HDFStore('../7D_nions0_flat.h5')
    store = pd.HDFStore('../7D_nions0_flat.h5')
    input = store['megarun1/input']
    data = store['megarun1/flattened']
    if 'megarun1/synthetic' in store:
        data = data.join(store['megarun1/synthetic'])
    #data = data.join(store['megarun1/combo'])
    #slicedim, style, nn_list = populate_nn_list(nn_set)
    slicedim, style, nns = nns_from_NNDB()
    #slicedim, style, nns = nns_from_manual()

    #nns = nns_from_nn_list(nn_list, slicedim, labels=labels)

    if style != 'similar':
        labels=True
    else:
        labels=False

    if mode == 'quick':
        filter_geq = -np.inf
        filter_less = np.inf
    else:
        filter_geq = -120
        filter_less = 120
    df, target_names = prep_df(input, data, nns, filter_less=filter_less, filter_geq=filter_geq)
    unsafe = is_unsafe(df, nns)
    if not unsafe:
        print('Warning! Cannot use unsafe mode')

    settings = mode_to_settings(mode)
    if settings['parallel']:
        num_processes = cpu_count()
        chunk_size = int(df.shape[0]/num_processes)
        chunks = [df.ix[df.index[i:i + chunk_size]] for i in range(0, df.shape[0], chunk_size)]
        pool = Pool(processes=num_processes)

    print('Starting {:d} slices for {:d} networks'.format(len(df), len(nns)))
    starttime = time.time()

    if not settings['parallel']:
        results = process_chunk(target_names, df, settings=settings, unsafe=unsafe)
    else:
        results = pool.map(partial(process_chunk, target_names, settings=settings, unsafe=unsafe), chunks)
    #for row in df.iterrows():
    #    process_row(row)
    print(len(df), 'took ', time.time() - starttime, ' seconds')

    zero_slices = 0
    totstats = []
    qlk_thresh = []
    for result in chain(*results):
        if result[0] == 1:
            zero_slices += 1
        else:
            totstats.append(result[2])
            qlk_thresh.append(result[1])

    stats = ['thresh', 'pop']
    totstats = pd.DataFrame(totstats, columns=pd.MultiIndex.from_tuples(list(product([nn.label for nn in nns.values()], target_names, stats))))

    qlk_columns = list(product(['QLK'], target_names, stats))
    qlk_data = np.full([len(totstats), len(qlk_columns)], np.nan)
    qlk_data[:, ::] = np.tile(qlk_thresh, np.array([len(qlk_columns),1])).T
    qlk_data = pd.DataFrame(qlk_data, columns=pd.MultiIndex.from_tuples(qlk_columns))

    totstats = totstats.join(qlk_data)
    res, duo_res = extract_stats(totstats, style)
    extract_nn_stats(res, duo_res, nns, submit_to_nndb=submit_to_nndb)


    #print('WARNING! If you continue, you will overwrite ', 'totstats_' + style + '.pkl')
    #embed()
    #totstats._metadata = {'zero_slices': zero_slices}
    #with open('totstats_' + style + '.pkl', 'wb') as file_:
    #    pickle.dump(totstats, file_)
