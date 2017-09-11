from IPython import embed
from multiprocessing import Pool
#import mega_nn
import numpy as np
import scipy.stats as stats
import pandas as pd
from itertools import product
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
from model import Network, NetworkJSON
from run_model import QuaLiKizNDNN
from train_NDNN import shuffle_panda

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
#    for varname, var in slice['input'].items():
#        try:
#            in_df = in_df.loc[np.isclose(in_df[varname], var, atol=1e-5, rtol=1e-3)]
#        except KeyError:
#            pass
nn_indeces = [37, 58, 60] #nozero <60, zero <60, zero <100
nn_indeces = [62, 63] #nozero mabse <60, zero mabse <60
nn_indeces = [58, 63]
from collections import OrderedDict
style = 'best'
mode = 'debug'
mode = 'quick'
if mode == 'debug':
    plot=True
    plot_pop=True
    plot_nns=True
    plot_slice=True
    plot_poplines=True
    plot_threshlines=True
    plot_zerocolors=False
    plot_thresh1line=False
    calc_thresh1=False
    hide_qualikiz=False
    debug=False
if mode == 'quick':
    plot=False
    plot_pop=False
    plot_nns=False
    plot_slice=False
    plot_poplines=False
    plot_threshlines=False
    plot_zerocolors=False
    plot_thresh1line=False
    calc_thresh1=False
    hide_qualikiz=False
    debug=False
if style == 'c_L2':
    nn_list = OrderedDict([(61, '$c_{L2} = 0.0$'),
    #                       (48, '$c_{L2} = 0.05$'),
                           (37, '$c_{L2} = 0.1$'),
    #                       (50, '$c_{L2} = 0.2$'),
    #                       (51, '$c_{L2} = 0.35$'),
                           (49, '$c_{L2} = 0.5$'),
    #                       (52, '$c_{L2} = 1.0$'),
                           (53, '$c_{L2} = 2.0$')])
elif style == 'topo':
    nn_list = OrderedDict([(65, 'neurons = $(10, 10)$'),
                           (64, 'neurons = $(30, 30)$'),
                           (73, 'neurons = $(30, 30, 30)$'),
                           (83, 'neurons = $(45, 45)$'),
                           (34, 'neurons = $(60, 60)$'),
                           (38, 'neurons = $(80, 80)$'),
                           (66, 'neurons = $(120, 120)$')])
elif style == 'filter':
    #nn_list = OrderedDict([(37, 'filter = 3'),
    #                       (58, 'filter = 4'),
    #                       (60, 'filter = 5')])
    nn_list = OrderedDict([(37, '$max(\chi_{ETG,e}) = 60$'),
                           (60, '$max(\chi_{ETG,e}) = 100$')])
elif style == 'goodness':
    nn_list = OrderedDict([(62, 'goodness = mabse'),
                           (37, 'goodness = mse')])
elif style == 'early_stop':
    nn_list = OrderedDict([(37, 'stop measure = loss'),
                           #(11, '$early_stop = mse'),
                           (18, 'stop measure = MSE')])
elif style == 'similar':
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
elif style == 'best':
    nn_list = OrderedDict([(46, '')])

#nn_list = OrderedDict([(88, 'efiITG_GB')])

nns = OrderedDict()
for nn_index, nn_label in nn_list.items():
    nn = nns[nn_index] = load_nn(nn_index)
    if style != 'similar':
        nn.label = nn_label
    else:
        nn.label = ''

input, data, __ = load_data(nn_index)
#input, data = prettify_df(input, data)
#input = input.astype('float64')
# Filter
varname = 'Ate'
#itor = zip(['An', 'Ati', 'Ti_Te', 'qx', 'smag', 'x'], ['1.00', '6.50', '2.50', '3.00', '-1.00', '0.09']); varname = 'Ate'
#for name, val in itor:
#    input = input[np.isclose(input[name], float(val),     atol=1e-5, rtol=1e-3)]

#input['x'] = input['x'] / 3
df = input.join([data['target'], data['maxgam']])
df = df[df['target']<60]
df = df[df['target']>=0]

#print(np.sum(df['target'] < 0)/len(df), ' frac < 0')
#print(np.sum(df['target'] == 0)/len(df), ' frac == 0')
#print(np.sum(df['target'] > 0)/len(df), ' frac > 0')
#uni = {col: input[col].unique() for col in input}
#uni_len = {key: len(value) for key, value in uni.items()}
#input['index'] = input.index
df.set_index([col for col in input], inplace=True)
df = df.astype('float64')
df = df.sort_index(level=varname)
df = df.unstack(varname)
#df = shuffle_panda(df)
#df.sort_values('smag', inplace=True)

sliced = 0
totstats = []
df = df.iloc[1040:20040,:]
# Check if we can do unsafe
unsafe = True
for nn in nns.values():
    varname_idx = nn._feature_names[nn._feature_names == varname].index[0]
    varlist = list(df.index.names)
    varlist.insert(varname_idx, varname)
    if ~np.all(varlist == nn._feature_names):
        unsafe = False

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
    try:
        idx = np.where(target == 0)[0][-1]
        #idx = target.index[target == 0][-1] #index of last zero
        #idx2 = feature[feature > idx][0]
        thresh2 = np.mean(feature[idx:idx+2])
    except IndexError:
        thresh2 = np.NaN
        if debug:
            print('No threshold2')

    return thresh2
#5.4 ms ± 115 µs per loop (mean ± std. dev. of 7 runs, 100 loops each) total
def process_row(row, ax1=None):
    index, slice = row
    target = slice['target']
    if np.all(target == 0):
        return (1,)
    else:
        # 156 µs ± 10.4 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each) (no zerocolors)
        thresh_nn = np.empty(len(nns))
        popbacks = np.empty(len(nns))
        thresh1_misses = np.empty(len(nns))
        thresh2_misses = np.empty(len(nns))
        if plot_zerocolors:
            maxgam = slice['maxgam']
        feature = slice['target'].index

        # Create slice, assume sorted
        # 14.8 µs ± 1.27 µs per loop (mean ± std. dev. of 7 runs, 100000 loops each)
        x = np.linspace(feature.values[0],
                        feature.values[-1],
                        100)
        #if plot:
        if not ax1 and plot:
            fig = plt.figure()
            if plot_pop and plot_slice:
                gs = gridspec.GridSpec(2, 2, height_ratios=[10, 1], width_ratios=[5,1],
                                    left=0.05, right=0.95, wspace=0.05, hspace=0.05)
                ax2 = plt.subplot(gs[1,0])
                ax3 = plt.subplot(gs[0,1])
            if not plot_pop and plot_slice:
                gs = gridspec.GridSpec(2, 1, height_ratios=[10, 2], width_ratios=[1],
                                    left=0.05, right=0.95, wspace=0.05, hspace=0.05)
                ax2 = plt.subplot(gs[1,0])
            if not plot_pop and not plot_slice:
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
            ax1.set_xlabel(nameconvert[varname])
            ax1.set_ylabel(nameconvert[list(nns.items())[0][1]._target_names[0]])
        if calc_thresh1:
            thresh1 = calculate_thresh1(x, feature, target, debug=debug)
            print('whyyy?')

        # 12.5 µs ± 970 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
        thresh2 = calculate_thresh2(feature.values, target.values, debug=debug)

        if plot and plot_threshlines:
            ax1.axvline(thresh2, c='black', linestyle='dashed')

        # 13.7 µs ± 1.1 µs per loop (mean ± std. dev. of 7 runs, 100000 loops each)
        if unsafe:
            slice_list = [np.full_like(x, val) for val in index]
            slice_list.insert(varname_idx, x)
        else:
            slice_dict = {name: np.full_like(x, val) for name, val in zip(df.index.names, index)}
            slice_dict[varname] = x



        # Plot target points

        if plot and plot_slice:
            table = ax2.table(cellText=[[nameconvert[name] for name in df.index.names],
                                        ['{:.2f}'.format(xx) for xx in index]],cellLoc='center')
            table.auto_set_font_size(False)
            table.scale(1, 1.5)
            #table.set_fontsize(20)
            ax2.axis('tight')
            ax2.axis('off')
        #fig.subplots_adjust(bottom=0.2, transform=ax1.transAxes)


        # Plot nn lines
        for ii, (nn_index, nn) in enumerate(nns.items()):
            if unsafe:
                nn_pred = nn.get_output(np.array(slice_list).T, safe=not unsafe, output_pandas=False)[:,0]
            else:
                nn_pred = nn.get_output(pd.DataFrame(slice_dict), safe=not unsafe, output_pandas=True).values[:,0]
            if plot and plot_nns:
                l = ax1.plot(x, nn_pred, label=nn.label)
            try:
                thresh_i = np.where(nn_pred == 0)[0][-1]
            except IndexError:
                thresh_nn[ii] = np.NaN
                if debug:
                    print('No threshold for network ', nn_index)
            else:
                thresh = thresh_nn[ii] = x[thresh_i]
                if plot and plot_threshlines:
                    ax1.axvline(thresh, c=l[0].get_color(), linestyle='dotted')
                if debug:
                    print('network ', nn_index, 'threshold ', thresh)
                #popback = popbacks[ii] = x[nn_pred[nn_pred.index[nn_pred[:thresh_i] != 0]].index[-1] + 1]
                popback_i = np.flatnonzero(nn_pred[:thresh_i])
                if popback_i.size != 0:
                    popback = popbacks[ii] = x[popback_i[-1]]
                    if plot and plot_poplines:
                        ax1.axvline(popback, c=l[0].get_color(), linestyle='dashed')
                else:
                    popbacks[ii] = np.NaN
                #embed()

        # 5.16 µs ± 188 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
        thresh2_misses = thresh2 - thresh_nn
        thresh2_popback = thresh2 - popbacks
        slice_stats = np.array([thresh2_misses, thresh2_popback]).T


        if plot and plot_pop:
            slice_strings = np.array(['{:.1f}'.format(xx) for xx in slice_stats.reshape(slice_stats.size)])
            slice_strings = slice_strings.reshape(slice_stats.shape)
            slice_strings = np.insert(slice_strings, 0, ['thre', 'pop'], axis=0)
            table = ax3.table(cellText=slice_strings, loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(20)
            ax3.axis('tight')
            ax3.axis('off')
        if debug:
            print(slice_stats.flatten())

        if plot:
            if plot_zerocolors:
                color = target.copy()
                color[(target == 0) & (maxgam == 0)] = 'green'
                color[(target != 0) & (maxgam == 0)] = 'red'
                color[(target == 0) & (maxgam != 0)] = 'magenta'
                color[(target != 0) & (maxgam != 0)] = 'blue'
            else:
                color='blue'
            if hide_qualikiz:
                color='white'
                zorder=1
                label=''
            else:
                zorder=1000
                label = 'Turbulence model'
                label=''
            ax1.scatter(feature, target, c=color, label=label, marker='x', zorder=zorder)

        # Plot regression
        if plot and plot_thresh1line and not np.isnan(thresh1):
            #plot_min = ax1.get_ylim()[0]
            plot_min = -0.1
            x_plot = x[(thresh_pred > plot_min) & (thresh_pred < ax1.get_ylim()[1])]
            y_plot = thresh_pred[(thresh_pred > plot_min) & (thresh_pred < ax1.get_ylim()[1])]
            ax1.plot(x_plot, y_plot, c='gray', linestyle='dotted')
            ax1.plot(x[x< thresh1], np.zeros_like(x[x< thresh1]), c='gray', linestyle='dotted')
            #ax1.axvline(thresh1, c='black', linestyle='dotted')

        if plot:
            ax1.legend()
            ax1.set_ylim(bottom=min(ax1.get_ylim()[0], 0))
            plt.show()
            fig.savefig('slice.pdf', format='pdf', bbox_inches='tight')
        return (0, slice_stats.flatten())
    #sliced += 1
    #if sliced % 1000 == 0:
    #    print(sliced, 'took ', time.time() - starttime, ' seconds')

pool = Pool(processes=4)
starttime = time.time()
res = pool.map_async(process_row, df.iterrows())
res = res.get()
#for row in df.iterrows():
#    process_row(row)
print(len(df), 'took ', time.time() - starttime, ' seconds')
embed()
#for index, slice in df.iterrows():
#    process_slice(slice)
#    #slice = slice.stack().reset_index(varname)
#    #slice = df.iloc[ii]

totstats =  pd.DataFrame(totstats, columns=pd.MultiIndex.from_tuples(list(product([nn.label for nn in nns.values()], ['thresh', 'pop']))))
print(sliced)
print('took ', time.time() - starttime, ' seconds')
#slice = df.sample(1)
#plt.scatter(slice[varname], target)

#for el in product(*uni.values()):
print('WARNING! If you continue, you will overwrite ', 'totstats_' + style + '.pkl')
embed()
totstats._metadata = {'zero_slices': zero_slices}
with open('totstats_' + style + '.pkl', 'wb') as file_:
    pickle.dump(totstats, file_)
