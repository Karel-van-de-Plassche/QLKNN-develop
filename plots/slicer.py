from IPython import embed
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

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec, cycler
from load_data import load_data, load_nn
#    for varname, var in slice['input'].items():
#        try:
#            in_df = in_df.loc[np.isclose(in_df[varname], var, atol=1e-5, rtol=1e-3)]
#        except KeyError:
#            pass
nn_indeces = [37, 58, 60] #nozero <60, zero <60, zero <100
nn_indeces = [62, 63] #nozero mabse <60, zero mabse <60
nn_indeces = [58, 63]
from collections import OrderedDict
style = 'early_stop'
plot=True
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
    nn_list = OrderedDict([(65, '$topo = (10, 10)$'),
                           (64, '$topo = (30, 30)$'),
                           (73, '$topo = (30, 30, 30)$'),
                           (83, '$topo = (45, 45)$'),
                           (34, '$topo = (60, 60)$'),
                           (38, '$topo = (80, 80)$'),
                           (66, '$topo = (120, 120)$')])
elif style == 'filter':
    nn_list = OrderedDict([(37, 'filter = 3'),
                           (58, 'filter = 4'),
                           (60, 'filter = 5')])
elif style == 'goodness':
    nn_list = OrderedDict([(62, 'goodness = mabse'),
                           (37, 'goodness = mse')])
elif style == 'early_stop':
    nn_list = OrderedDict([(37, 'early_stop = loss'),
                           #(11, '$early_stop = mse'),
                           (18, 'early_stop = mse')])
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

nns = OrderedDict()
for nn_index, nn_label in nn_list.items():
    nn = nns[nn_index] = load_nn(nn_index)
    nn.label = nn_label

input, data, __ = load_data(nn_index)
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
varname = 'Ate'
df = df.unstack(varname)
df = shuffle_panda(df)

sliced = 0
starttime = time.time()
zero_slices = 0
thresh_nn = np.empty(len(nns))
popbacks = np.empty(len(nns))
thresh1_misses = np.empty(len(nns))
thresh2_misses = np.empty(len(nns))
totstats = []
for index, slice in df.iterrows():
    #slice = slice.stack().reset_index(varname)
    #slice = df.iloc[ii]
    target = slice['target']
    maxgam = slice['maxgam']
    feature = slice['target'].index
    if np.all(target == 0):
        zero_slices += 1
    else:
        x = np.linspace(feature.min(),
                        feature.max(),
                        200)
        if plot:
            fig = plt.figure()
            gs = gridspec.GridSpec(2, 2, height_ratios=[10, 1], width_ratios=[5,1],
                                   left=0.05, right=0.95, wspace=0.05, hspace=0.05)
            ax1 = plt.subplot(gs[0,0])
            #ax1.set_prop_cycle(cycler('color', ['#f1eef6','#d7b5d8','#df65b0','#dd1c77','#980043']))
            # http://tristen.ca/hcl-picker/#/clh/5/273/2A0A75/D59FEB
            #ax1.set_prop_cycle(cycler('color', ['#2A0A75','#6330B8','#9F63E2','#D59FEB']))
            ax1.set_prop_cycle(cycler('color', plt.cm.plasma(np.linspace(0, 0.9, len(nns)))))
            ax2 = plt.subplot(gs[1,0])
            ax3 = plt.subplot(gs[0,1])

        #try:
        #    idx = target.index[target == 0][-1] #index of last zero
        #    slope, intercept, r_value, p_value, std_err = stats.linregress(feature[(target.index > idx) & ~target.isnull()], target[(target.index > idx) & ~target.isnull()])
        #    thresh_pred = x * slope + intercept
        #    thresh1 = x[thresh_pred < 0][-1]

        #    if plot:
        #        #ax1.plot(x[(thresh_pred > ax1.get_ylim()[0]) & (thresh_pred < ax1.get_ylim()[1])],
        #        #         thresh_pred[(thresh_pred > ax1.get_ylim()[0]) & (thresh_pred < ax1.get_ylim()[1])],
        #        #         c='black')
        #        ax1.axvline(thresh1, c='black', linestyle='dotted')
        #except (ValueError, IndexError):
        #    thresh1 = np.NaN
        #    if debug:
        #        print('No threshold1')
        try:
            idx = target.index[target == 0][-1] #index of last zero
            idx2 = feature[feature > idx][0]
            thresh2 = np.mean([idx2, idx])
            if plot:
                ax1.axvline(thresh2, c='black', linestyle='dashed')
        except IndexError:
            thresh2 = np.NaN
            if debug:
                print('No threshold2')
        slice_dict = {name: np.full_like(x, val) for name, val in zip(df.index.names, index)}
        slice_dict[varname] = x

        # Plot target points
        if plot:
            color = target.copy()
            color[(target == 0) & (maxgam == 0)] = 'green'
            color[(target != 0) & (maxgam == 0)] = 'red'
            color[(target == 0) & (maxgam != 0)] = 'magenta'
            color[(target != 0) & (maxgam != 0)] = 'blue'
            ax1.scatter(feature, target, c=color)

        if plot:
            table = ax2.table(cellText=[df.index.names, ['{:.2f}'.format(xx) for xx in index]])
            table.auto_set_font_size(False)
            table.set_fontsize(20)
            ax2.axis('tight')
            ax2.axis('off')
        #fig.subplots_adjust(bottom=0.2, transform=ax1.transAxes)


        # Plot nn lines
        for ii, (nn_index, nn) in enumerate(nns.items()):
            nn_pred = nn.get_output(**slice_dict).iloc[:,0]
            if plot:
                l = ax1.plot(x, nn_pred, label=nn.label)
            try:
                thresh_i = nn_pred.index[nn_pred == 0][-1]
            except IndexError:
                thresh_nn[ii] = np.NaN
                if debug:
                    print('No threshold for network ', nn_index)
            else:
                thresh = thresh_nn[ii] = x[thresh_i]
                if plot:
                    ax1.axvline(thresh, c=l[0].get_color(), linestyle='dotted')
                if debug:
                    print('network ', nn_index, 'threshold ', thresh)
                try:
                    popback = popbacks[ii] = x[nn_pred[nn_pred.index[nn_pred[:thresh_i] != 0]].index[-1] + 1]
                    if plot:
                        ax1.axvline(popback, c=l[0].get_color(), linestyle='dashed')
                except IndexError:
                    popbacks[ii] = np.NaN

        thresh2_misses = thresh2 - thresh_nn
        thresh2_popback = thresh2 - popbacks

        slice_stats = np.array([thresh2_misses, thresh2_popback]).T
        if plot:
            slice_strings = np.array(['{:.1f}'.format(xx) for xx in slice_stats.reshape(slice_stats.size)])
            slice_strings = slice_strings.reshape(slice_stats.shape)
            slice_strings = np.insert(slice_strings, 0, ['thre', 'pop'], axis=0)
            table = ax3.table(cellText=slice_strings, loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(20)
            ax3.axis('tight')
            ax3.axis('off')
        totstats.append(slice_stats.flatten())
        if debug:
            print(slice_stats.flatten())

        # Plot regression
        if plot:
            ax1.legend()
            plt.show()
    sliced += 1
    if sliced % 1000 == 0:
        print(sliced, 'took ', time.time() - starttime, ' seconds')

totstats =  pd.DataFrame(totstats, columns=pd.MultiIndex.from_tuples(list(product([nn.label for nn in nns.values()], ['thresh', 'pop']))))
print(sliced)
print(zero_slices)
print('took ', time.time() - starttime, ' seconds')
#slice = df.sample(1)
#plt.scatter(slice[varname], target)

#for el in product(*uni.values()):
print('WARNING! If you continue, you will overwrite ', 'totstats_' + style + '.pkl')
embed()
totstats._metadata = {'zero_slices': zero_slices}
with open('totstats_' + style + '.pkl', 'wb') as file_:
    pickle.dump(totstats, file_)
