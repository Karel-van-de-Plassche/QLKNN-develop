from IPython import embed
#import mega_nn
import gc
import numpy as np
import pandas as pd
from itertools import chain
from model import Network, NetworkJSON, Hyperparameters, Postprocessing
from peewee import Param
import os
import sys
networks_path = os.path.abspath(os.path.join((os.path.abspath(__file__)), '../../networks'))
sys.path.append(networks_path)
from run_model import QuaLiKizNDNN


target_to_fancy = {'efeETG_GB': 'Electron ETG Heat Flux',
                   'efeITG_GB': 'Electron ITG Heat Flux',
                   'efeTEM_GB': 'Electron TEM Heat Flux',
                   'efiITG_GB': 'Ion ITG Heat Flux',
                   'efiTEM_GB': 'Ion TEM Heat Flux'}
feature_names = ['An','Ate','Ati','Ti_Te','qx','smag','x']
query = (Network.select(Network.target_names).distinct().tuples())
df = pd.DataFrame(columns=['target_names', 'id', 'l2_norm', 'rms', 'rms_rel'])
for ii, query_res in enumerate(query):
    target_names = query_res[0]
    if len(target_names) == 1:
        target_name = target_names[0]
    else:
        NotImplementedError('Multiple targets not implemented yet')
    print(target_name)
    subquery = (Network.select(Network.id, Postprocessing.filtered_loss, Postprocessing.filtered_rms, 100*Postprocessing.rel_filtered_rms)
                .where(Network.target_names == Param(target_names))
                .where(Network.feature_names == Param(feature_names))
                .join(Postprocessing)
                .order_by(Postprocessing.filtered_real_loss)
                .limit(1)
                .tuples())
    df.loc[ii] = list(chain([target_name], subquery.get()))
df['id'] = df['id'].astype('int64')

print(df)
for row in df.iterrows():
    df.set_value(row[0], 'target_names', target_to_fancy[row[1]['target_names']])
df = df[['target_names', 'l2_norm', 'rms', 'rms_rel']]
df.columns = ['Training Target', 'L_2 norm', 'RMS error [GB]', 'RMS error [%]']
print(df.to_latex(index=False))
embed()
