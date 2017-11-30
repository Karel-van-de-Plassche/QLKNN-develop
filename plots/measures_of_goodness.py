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
import model
from model import Network, NetworkJSON, Hyperparameters, PostprocessSlice, NetworkMetadata, TrainMetadata, ComboNetwork, MultiNetwork, Postprocess
from run_model import QuaLiKizNDNN
#from train_NDNN import shuffle_panda

from peewee import Param, Passthrough, JOIN, prefetch

import matplotlib.pyplot as plt
from matplotlib import gridspec
from load_data import load_data, load_nn

target_names = ['efiITG_GB', 'efeITG_GB']
query = (Network.select(Network.id,
                        Hyperparameters.hidden_neurons,
                        PostprocessSlice,
                        Postprocess.rms,
                        TrainMetadata.l2_norm
         )
         .join(Hyperparameters,  on=(Network.id == Hyperparameters.network_id))
         .join(NetworkMetadata,  on=(Network.id == NetworkMetadata.network_id))
         .join(TrainMetadata,    on=(Network.id == TrainMetadata.network_id))
         .join(PostprocessSlice, on=(Network.id == PostprocessSlice.network_id))
         .join(Postprocess, on=(Network.id == Postprocess.network_id))
         .where(Network.target_names == Param(target_names))
         .where(TrainMetadata.set == 'train')
         )
results = list(query.dicts())
df = pd.DataFrame(results)
df.drop(['id', 'multi_network', 'combo_network'], inplace=True, axis='columns')
df['network'] = df['network'].apply(lambda el: 'pure_' + str(el))
df['l2_norm'] = df['l2_norm'].apply(np.nanmean)
df.set_index('network', inplace=True)
stats = df

query = (MultiNetwork.select(MultiNetwork.id,
                             PostprocessSlice,
                             Postprocess.rms)
         .join(PostprocessSlice, on=(MultiNetwork.id == PostprocessSlice.multi_network_id))
         .join(Postprocess, on=(MultiNetwork.id == Postprocess.multi_network_id))
         .where(MultiNetwork.target_names == Param(target_names))
)
results = list(query.dicts())
df = pd.DataFrame(results)
df.drop(['id', 'network', 'combo_network'], inplace=True, axis='columns')
df['multi_network'] = df['multi_network'].apply(lambda el: 'multi_' + str(el))
df.rename(columns = {'multi_network':'network'}, inplace = True)
df.set_index('network', inplace=True)
stats = pd.concat([stats, df])

stats = stats.applymap(np.array)
#stats[stats.isnull()] = np.NaN
stats.sort_index(inplace=True)
embed()
stats['hidden_neurons_total'] = stats.pop('hidden_neurons').to_frame().applymap(np.sum)
stats['l2_norm_weighted'] = stats.pop('l2_norm') / stats.pop('hidden_neurons_total')
stats.dropna(axis='columns', how='all', inplace=True)
#'no_pop_frac', 'no_thresh_frac', 'pop_abs_mis_95width',
#       'pop_abs_mis_median', 'rms_test', 'thresh_rel_mis_95width',
#       'thresh_rel_mis_median', 'l2_norm_weighted'
#print(stats.max())
#print(stats.min())
#print(stats.mean())
#print(stats.abs().mean())
del stats['pop_abs_mis_95width']
del stats['thresh_rel_mis_95width']
del stats['dual_thresh_mismatch_95width']
stats['rms'] = stats.pop('rms')
stats['thresh'] = stats.pop('thresh_rel_mis_median').abs().apply(np.max)
stats['no_thresh_frac'] = stats.pop('no_thresh_frac').apply(np.max)
stats['pop'] = (14 - stats.pop('pop_abs_mis_median').abs()).apply(np.max)
stats['l2'] = stats.pop('l2_norm_weighted')
stats['pop_frac'] = (1 - stats.pop('no_pop_frac')).apply(np.max)
stats['thresh_mismatch'] = stats.pop('dual_thresh_mismatch_median').abs().apply(np.max)
#(stats/stats.max()).nsmallest(10, 'rms').plot.bar()

fig = plt.figure()
gs = gridspec.GridSpec(2, 1, height_ratios=[10, 2], width_ratios=[1],
                       left=0.05, right=0.95, wspace=0.05, hspace=0.05)
ax2 = plt.subplot(gs[1,0])
ax1 = plt.subplot(gs[0,0])
top = (stats).nsmallest(10, 'rms')
subplot = (top/top.max()).plot.bar(ax=ax1)
text = [(col, '{:.2f}'.format(top[col].max())) for col in top]
text = list(map(list, zip(*text))) #Transpose
embed()
table = ax2.table(cellText=text, cellLoc='center')
table.auto_set_font_size(False)
table.scale(1, 1.5)
#table.set_fontsize(20)
ax2.axis('tight')
ax2.axis('off')
#(np.log10(stats/stats.max())).loc[stats.sum(axis='columns').nsmallest(10).index].plot.bar()
plt.show()
