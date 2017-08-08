from IPython import embed
#import mega_nn
import numpy as np
import scipy.stats as stats
import pandas as pd
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
from load_data import load_data, load_nn
#    for varname, var in slice['input'].items():
#        try:
#            in_df = in_df.loc[np.isclose(in_df[varname], var, atol=1e-5, rtol=1e-3)]
#        except KeyError:
#            pass
nn_indeces = [46, 58, 60] #nozero <60, zero <60, zero <100
input, df, __ = load_data(nn_indeces[0])
df = input.join(df['target'])
df = df[df['target']<60]
df = df[df['target']>=0]
nns = {}
for nn_index in nn_indeces:
    nns[nn_index] = load_nn(nn_index)

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
for index, slice in df.iterrows():
    #slice = slice.stack().reset_index(varname)
    #slice = df.iloc[ii]
    target = slice['target']
    feature = slice['target'].index
    if np.all(target == 0):
        zero_slices += 1
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        x = np.linspace(feature.min(),
                        feature.max(),
                        200)
        slice_dict = {name: np.full_like(x, val) for name, val in zip(df.index.names, index)}
        slice_dict[varname] = x

        # Plot target points
        color = target.copy()
        color[target == 0] = 'r'
        color[target != 0] = 'b'
        ax.scatter(feature, target, c=color)

        # Plot nn lines
        for nn_index, nn in nns.items():
            nn_pred = nn.get_output(**slice_dict)
            l = ax.plot(x, nn_pred, label=nn_index)
            try:
                thresh = x[nn_pred.index[nn_pred.iloc[:,0] == 0][-1]]
                ax.axvline(thresh, c=l[0].get_color(), linestyle='dotted')
                print('network ', nn_index, 'threshold ', thresh)
            except:
                print('No threshold for network ', nn_index)
            else:
                thresh = None

        # Plot regression
        try:
            idx = target.index[target == 0][-1] #index of last zero
            slope, intercept, r_value, p_value, std_err = stats.linregress(feature[(target.index > idx) & ~target.isnull()], target[(target.index > idx) & ~target.isnull()])
            thresh_pred = x * slope + intercept
            thresh_0 = x[thresh_pred < 0][-1]

            ax.plot(x[(thresh_pred > ax.get_ylim()[0]) & (thresh_pred < ax.get_ylim()[1])],
                    thresh_pred[(thresh_pred > ax.get_ylim()[0]) & (thresh_pred < ax.get_ylim()[1])],
                    c='black')
            ax.axvline(thresh_0, c='black', linestyle='dotted')
        except:
            print('No threshold')
        try:
            idx = target.index[target == 0][-1] #index of last zero
            print(idx)
            idx2 = feature[feature > idx][0]
            print(idx2)
            ax.axvline(np.mean([idx2, idx]), c='black', linestyle='dashed')
        except:
            print('No threshold2')
        ax.legend()
        plt.show()
    sliced += 1
    if sliced > 1000:
        break
print(sliced)
print(zero_slices)
print('took ', time.time() - starttime, ' seconds')
#slice = df.sample(1)
#plt.scatter(slice[varname], target)

#for el in product(*uni.values()):
embed()
