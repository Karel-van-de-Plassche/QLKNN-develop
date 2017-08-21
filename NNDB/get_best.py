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


feature_names = ['An','Ate','Ati','Ti_Te','qx','smag','x']
query = (Network.select(Network.target_names).distinct().tuples())
df = pd.DataFrame(columns=['target_names', 'id', 'l2_norm', 'rms'])
for ii, query_res in enumerate(query):
    target_names = query_res[0]
    if len(target_names) == 1:
        target_name = target_names[0]
    else:
        NotImplementedError('Multiple targets not implemented yet')
    print(target_name)
    subquery = (Network.select(Network.id, Postprocessing.filtered_loss, Postprocessing.filtered_rms)
                .where(Network.target_names == Param(target_names))
                .where(Network.feature_names == Param(feature_names))
                .join(Postprocessing)
                .order_by(Postprocessing.filtered_real_loss)
                .limit(1)
                .tuples())
    df.loc[ii] = list(chain([target_name], subquery.get()))
df['id'] = df['id'].astype('int64')
embed()
