import re

import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from peewee import AsIs, JOIN, prefetch, SQL
from IPython import embed

from qlknn.NNDB.model import Network, PureNetworkParams, PostprocessSlice, NetworkMetadata, TrainMetadata, Postprocess, db

def deconstruct_varname(varname):
    flux, species, regime, normalization, _ = re.compile('(?=.*)(.)(|ITG|ETG|TEM)_(GB|SI|cm)').split(varname)
    return flux, species, regime, normalization

target_names = ['efeITG_GB', 'efiITG_GB', 'pfeITG_GB']
__, __, regime, norm = deconstruct_varname(target_names[0])
leading = {'ITG': 'efe',
           'TEM': 'efi',
           'ETG': 'efe'}
leading_flux = leading[regime] + regime + '_' + norm
leading_idx = target_names.index(leading_flux)

query = (Network.select(Network.id.alias('network_id'),
                        PostprocessSlice,
                        Postprocess.rms,
         )
         .join(PostprocessSlice, JOIN.LEFT_OUTER)
         .switch(Network)
         .join(Postprocess, JOIN.LEFT_OUTER)
         .switch(Network)
         .where(Network.target_names == target_names)
         #.join(PureNetworkParams)
         )
         #.join(Hyperparameters,  on=(Network.id == Hyperparameters.network_id))
         #.join(NetworkMetadata,  on=(Network.id == NetworkMetadata.network_id))
         #.join(TrainMetadata,    on=(Network.id == TrainMetadata.network_id))
         #.join(PostprocessSlice, on=(Network.id == PostprocessSlice.network_id))
         #.join(Postprocess, on=(Network.id == Postprocess.network_id))
         #.where(Network.target_names == target_names)
         #.where(TrainMetadata.set == 'train')
         #.where((PostprocessSlice.dual_thresh_mismatch_median == 0) | PostprocessSlice.dual_thresh_mismatch_median.is_null())
         #)
if query.count() > 0:
    results = list(query.dicts())
    df = pd.DataFrame(results)
    #df['network'] = df['network'].apply(lambda el: 'pure_' + str(el))
    #df['l2_norm'] = df['l2_norm'].apply(np.nanmean)
    df.drop(['id', 'network'], inplace=True, axis='columns')
    df.set_index('network_id', inplace=True)
    stats = df
else:
    stats = pd.DataFrame()

for net_id in stats.index:
    res_dict = {}
    for param in ['cost_l2_scale', 'hidden_neurons']:
        res = Network.get_by_id(net_id).flat_recursive_property(param)
        if isinstance(res, np.ndarray):
            res = res.astype(object)
        df.set_value(net_id, param, res)

stats = stats.applymap(np.array)
#stats[stats.isnull()] = np.NaN
stats.sort_index(inplace=True)
stats.dropna(axis='columns', how='all', inplace=True)
#'no_pop_frac', 'no_thresh_frac', 'pop_abs_mis_95width',
#       'pop_abs_mis_median', 'rms_test', 'thresh_rel_mis_95width',
#       'thresh_rel_mis_median', 'l2_norm_weighted'
#print(stats.max())
#print(stats.min())
#print(stats.mean())
#print(stats.abs().mean())
dont_care = ['frac', 'no_dual_thresh_frac', 'no_pop_frac', 'no_thresh_frac', 'pop_abs_mis_95width', 'thresh_rel_mis_95width', 'wobble_tot', 'wobble_unstab']
stats.drop(dont_care, axis='columns', inplace=True, errors='ignore')
array_care = ['pop_abs_mis_median', 'thresh_rel_mis_median', 'wobble_qlkunstab', 'rms']
for var in array_care:
    stats[var] = stats.agg({var: lambda x: x[leading_idx]})
embed()

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
