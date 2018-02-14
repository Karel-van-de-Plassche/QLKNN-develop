from IPython import embed
#import mega_nn
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
from model import Network, NetworkJSON
from run_model import QuaLiKizNDNN
from train_NDNN import shuffle_panda
from matplotlib.colors import LogNorm

import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
plt.style.use('./thesis.mplstyle')

from load_data import load_data

def zero_linregress(x, y, force_xy=True):
    x_ = x[:,np.newaxis]
    if force_xy:
        a = 1
    else:
        a, _, _, _ = np.linalg.lstsq(x_, y)

    f = a * x
    y_bar = np.sum(y) / len(y)
    ss_res = np.sum(np.square(y - f))
    ss_tot = np.sum(np.square(y - y_bar))
    r_value = np.sqrt(1 - ss_res / ss_tot)
    return a, r_value

##############################################################################
# Load Data                                                                  #
##############################################################################
_, df, __ = load_data(46)
store = pd.HDFStore('../filtered_7D_nions0_flat_filter3.h5')
index = store['index']
df = df.loc[index]

maxzoom = 40
##############################################################################
# Filter dataset                                                             #
##############################################################################
df = df[df['target']<maxzoom]
df = df[df['prediction']<df['target'].max()]
df = df[df['target']>0]
df = df[df['prediction']>0]

#df = df[df['target']>0.1]
#df = df[df['prediction']>0]
#df = df.loc[(df['residuals']**2).sort_values(ascending=False)[int(0.1 * len(df)):].index]
#df = shuffle_panda(df)[int(0.99 * len(df)):]

x = df['target']
y = df['prediction']
res = df['residuals']
style = 'nonlog'

fig = plt.figure()
ax = fig.add_subplot(111)
##############################################################################
# Plot heatmap                                                               #
##############################################################################
if style == 'log':
    heatmap, xedges, yedges = np.histogram2d(np.log10(x), np.log10(y), bins=120)
else:
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=120)
heatmap = heatmap.T
#heatmap = np.log10(heatmap.T)
minlog = 5 * 10**1
heatmap[heatmap < minlog] = 0
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

cax = ax.imshow(heatmap, extent=extent, interpolation='none', origin='lower', aspect='auto', 
                norm=LogNorm(vmin=minlog, vmax=np.max(heatmap)),
                cmap='inferno_r')
#cax = ax.matshow(heatmap, norm=LogNorm(vmin=0.01, vmax=np.max(heatmap)))
if style == 'log':
    ax.set_ylim(-1.6, np.log10(60))
    ax.set_xlim(-1.6, np.log10(60))
    ax.vlines(1, ax.get_ylim()[0], 1, linestyle='dotted')
    ax.hlines(1, ax.get_xlim()[0], 1, linestyle='dotted')
else:
    ax.set_ylim(0, maxzoom)
    ax.set_xlim(0, maxzoom)
ax.set_aspect('equal')
ax.set_xlabel('QuaLiKiz')
ax.set_ylabel('Neural Network')
cb = fig.colorbar(cax)
cb.set_label('counts')

##############################################################################
# Plot regression                                                            #
##############################################################################
slope, r_value = zero_linregress(x, y)
intercept = 0
x_reg = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 50)
#ax.plot(x_reg, slope * x_reg + intercept, c='black', linestyle='dashed')
ax.plot(x_reg, x_reg, c='black', linestyle='dashed')
#ax.text(.8,.9, "$R^2 = " + str(np.round(r_value**2, 3)) + '$', fontsize=15, transform=ax.transAxes)
print(r_value**2)

#import seaborn as sns
#sns.set(style="darkgrid", color_codes=True)

#g = sns.jointplot("target", "prediction", data=df, kind="reg",
#                  xlim=(0, 60), ylim=(0, 60), color="r", size=7)
#g = sns.lmplot("target", "prediction", data=df)
fig.savefig('rms_heatmap.pdf')

plt.show()
#embed()
