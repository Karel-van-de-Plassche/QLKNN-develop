from IPython import embed
#import mega_nn
import gc
import numpy as np
import pandas as pd
from model import Network, NetworkJSON, Hyperparameters, Postprocessing
from peewee import Param
import os
import sys
networks_path = os.path.abspath(os.path.join((os.path.abspath(__file__)), '../../networks'))
sys.path.append(networks_path)
from run_model import QuaLiKizNDNN


store_filtered = {}
store_filtered[3] = pd.HDFStore('../filtered_7D_nions0_flat_filter3.h5')
store_filtered[4] = pd.HDFStore('../filtered_7D_nions0_flat_filter4.h5')
store_filtered[5] = pd.HDFStore('../filtered_7D_nions0_flat_filter5.h5')
store = pd.HDFStore('../7D_nions0_flat.h5')
input = store['megarun1/input']
df = store['megarun1/flattened']

#nn = mega_nn.nn

root_name = '/megarun1/nndb_nn/'
query = (Network.select(Network.id, Network.target_names, Network.filter).tuples())
for query_res in query:
    id, target_names, filter_id = query_res

    try:
        Network.get(Network.id == id).postprocessing.get()
    except Postprocessing.DoesNotExist as e:
        pass
    else:
        continue

    if len(target_names) == 1:
        target_name = target_names[0]
    else:
        NotImplementedError('Multiple targets not implemented yet')
    print(target_name, id)
    parent_name = root_name + target_name + '/'
    network_name = parent_name + str(id) + '_noclip'
    #if network_name in store:
    #    print(network_name, 'not in store')
    #    continue
    df = store_filtered[filter_id]

    try:
        df_nn = store[network_name]

    except KeyError:
        continue

    subquery = (Network.select(Hyperparameters.cost_l1_scale,
                               Hyperparameters.cost_l2_scale,
                               Hyperparameters.goodness,
                               NetworkJSON.network_json)
                .where(Network.id == id)
                .join(Hyperparameters, on=Hyperparameters.network_id == Network.id)
                .join(NetworkJSON, on=NetworkJSON.network_id == Network.id)
                .tuples())
    cost_l1_scale, cost_l2_scale, goodness, json_dict = subquery.get()
    nn = QuaLiKizNDNN(json_dict)
    se = df[target_name]
    #se = se.loc[se > 0]
    #se = se.loc[se < 90]

    #df_nn = nn.get_output(**input.loc[se.index])
    #df_nn.index = se.index
    se_nn = df_nn[target_name].astype('float64').loc[df.index]
    high_bound = nn.target_max[target_name]
    low_bound = nn.target_min[target_name]
    se_nn[se_nn < low_bound] = low_bound
    se_nn[se_nn > high_bound] = high_bound
    #se_nn.index = df[target_name].index
    #se_nn = se_nn.loc[se_nn > nn.target_min[target_name]]
    #se_nn = se_nn.loc[se_nn < nn.target_max[target_name]]
    res = se_nn - se
    filtered_mse = np.mean(np.square(res))
    filtered_mabse = np.mean(np.abs(res))

    l1_norm = nn.l1_norm
    l2_norm = nn.l2_norm

    filtered_loss = 0
    if goodness == 'mse':
        filtered_loss = filtered_mse
    elif goodness == 'mabse':
        filtered_loss = filtered_mabse
    if cost_l1_scale != 0:
        filtered_loss += cost_l1_scale * l1_norm
    if cost_l2_scale != 0:
        filtered_loss += cost_l2_scale * l2_norm


    real_loss_function = 'filtered_mse + 0.1 * l2_norm'
    real_loss = eval(real_loss_function)

    pp = Postprocessing(network=Network.get(Network.id == id),
                        filtered_rms=np.sqrt(filtered_mse),
                        l2_norm=l2_norm,
                        filtered_loss=filtered_loss,
                        filtered_real_loss=real_loss,
                        filtered_real_loss_function=real_loss_function)
    pp.save()

