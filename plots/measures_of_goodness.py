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
from model import Network, NetworkJSON, Hyperparameters, PostprocessSlice, NetworkMetadata, TrainMetadata, ComboNetwork, MultiNetwork, Postprocess, db
from run_model import QuaLiKizNDNN
#from train_NDNN import shuffle_panda

from peewee import AsIs, JOIN, prefetch, SQL

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
         .where(Network.target_names == target_names)
         .where(TrainMetadata.set == 'train')
         .where((PostprocessSlice.dual_thresh_mismatch_median == 0) | PostprocessSlice.dual_thresh_mismatch_median.is_null())
         )
if query.count() > 0:
    results = list(query.dicts())
    df = pd.DataFrame(results)
    df.drop(['id', 'multi_network', 'combo_network'], inplace=True, axis='columns')
    df['network'] = df['network'].apply(lambda el: 'pure_' + str(el))
    df['l2_norm'] = df['l2_norm'].apply(np.nanmean)
    df.set_index('network', inplace=True)
    stats = df
else:
    stats = pd.DataFrame()

query = (ComboNetwork.select(ComboNetwork.id,
                             PostprocessSlice,
                             Postprocess.rms,
                             #SQL("'hidden_neurons'")
                             )
         .join(PostprocessSlice, on=(ComboNetwork.id == PostprocessSlice.combo_network_id))
         .join(Postprocess, on=(ComboNetwork.id == Postprocess.combo_network_id))
         .where(ComboNetwork.target_names == target_names)
         .where((PostprocessSlice.dual_thresh_mismatch_median == 0) | PostprocessSlice.dual_thresh_mismatch_median.is_null())
)
compound_stats = ['hidden_neurons', 'cost_l2_scale']
for compound_stat in compound_stats:
    subquery = ComboNetwork.calc_op(compound_stat).alias(compound_stat)
    query = query.select(SQL('*')).join(subquery, on=(ComboNetwork.id == subquery.c.combo_id))
#query3 = ComboNetwork.calc_op('hidden_neurons').alias('hidden_neurons')

ignore_list = ['network_id', 'multi_network_id', 'leq_bound', 'less_bound', 'id', 'frac', 'filter_id', 'feature_names', 'combo_network_id', 'target_names']
if query.count() > 0:
    results = list(query.dicts())
    df = pd.DataFrame(results)
    df.drop(ignore_list + ['recipe', 'networks'] , inplace=True, axis='columns')
    df['combo_id'] = df['combo_id'].apply(lambda el: 'combo_' + str(el))
    df.rename(columns = {'combo_id':'network'}, inplace = True)
    df.set_index('network', inplace=True)
    for compound_stat in compound_stats:
        df[compound_stat] = df[compound_stat].apply(lambda el: el[0] if el[1:] == el[:-1] else None)
    stats = pd.concat([stats, df])

query = (MultiNetwork.select(MultiNetwork.id,
                             PostprocessSlice,
                             Postprocess.rms)
         .join(PostprocessSlice, on=(MultiNetwork.id == PostprocessSlice.multi_network_id))
         .join(Postprocess, on=(MultiNetwork.id == Postprocess.multi_network_id))
         .where(MultiNetwork.target_names == target_names)
         .where((PostprocessSlice.dual_thresh_mismatch_median == 0) | PostprocessSlice.dual_thresh_mismatch_median.is_null())
)
for compound_stat in compound_stats:
    subquery = MultiNetwork.calc_op(compound_stat).alias(compound_stat)
    query = query.select(SQL('*')).join(subquery, on=(MultiNetwork.id == subquery.c.multi_id))

if query.count() > 0:
    results = list(query.dicts())
    df = pd.DataFrame(results)
    df.drop(ignore_list + ['network_partners', 'combo_network_partners'], inplace=True, axis='columns')
    df['multi_id'] = df['multi_id'].apply(lambda el: 'multi_' + str(el))
    df.rename(columns = {'multi_id':'network'}, inplace = True)
    for compound_stat in compound_stats:
        df[compound_stat] = df[compound_stat].apply(lambda el: el[0] if el[1:] == el[:-1] else None)
        df[compound_stat] = df[compound_stat].apply(lambda el: el[0] if el[1:] == el[:-1] else None)
    df.set_index('network', inplace=True)
    stats = pd.concat([stats, df])

stats = stats.applymap(np.array)
#stats[stats.isnull()] = np.NaN
stats.sort_index(inplace=True)
try:
    stats['hidden_neurons_total'] = stats.pop('hidden_neurons').to_frame().applymap(np.sum)
    stats['l2_norm_weighted'] = stats.pop('l2_norm') / stats.pop('hidden_neurons_total')
    stats['l2'] = stats.pop('l2_norm_weighted')
except KeyError:
    pass
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
stats['rms'] = stats.pop('rms')
stats['thresh'] = stats.pop('thresh_rel_mis_median').abs().apply(np.max)
stats['no_thresh_frac'] = stats.pop('no_thresh_frac').apply(np.max)
stats['pop'] = (14 - stats.pop('pop_abs_mis_median').abs()).apply(np.max)
if 'wobble_tot' in stats.keys():
    stats['wobble_tot'] = stats.pop('wobble_tot').apply(np.max)
    stats['wobble_unstab'] = stats.pop('wobble_unstab').apply(np.max)
stats['pop_frac'] = (1 - stats.pop('no_pop_frac')).apply(np.max)
#stats.dropna(inplace=True)
try:
    del stats['dual_thresh_mismatch_95width']
    stats['thresh_mismatch'] = stats.pop('dual_thresh_mismatch_median').abs().apply(np.max)
except KeyError:
    pass
#(stats/stats.max()).nsmallest(10, 'rms').plot.bar()

fig = plt.figure()
gs = gridspec.GridSpec(2, 1, height_ratios=[10, 2], width_ratios=[1],
                       left=0.05, right=0.95, wspace=0.05, hspace=0.05)
ax2 = plt.subplot(gs[1,0])
ax1 = plt.subplot(gs[0,0])
top = (stats).nsmallest(10, 'rms')
top.dropna('columns', inplace=True)
subplot = (top/top.max()).plot.bar(ax=ax1)
text = [(col, '{:.2f}'.format(top[col].max())) for col in top]
text = list(map(list, zip(*text))) #Transpose
table = ax2.table(cellText=text, cellLoc='center')
table.auto_set_font_size(False)
table.scale(1, 1.5)
#table.set_fontsize(20)
ax2.axis('tight')
ax2.axis('off')
#(np.log10(stats/stats.max())).loc[stats.sum(axis='columns').nsmallest(10).index].plot.bar()
#plt.show()
embed()
