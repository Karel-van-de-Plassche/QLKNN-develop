from IPython import embed
#import mega_nn
import numpy as np
import pandas as pd
from NNDB import Network, NetworkJSON
from peewee import Param
from run_model import QuaLiKizNDNN

store = pd.HDFStore('./7D_nions0_flat.h5')
input = store['megarun1/input']
df = store['megarun1/flattened']

#nn = mega_nn.nn

root_name = '/megarun1/nndb_nn/'
query = (Network.select(Network.target_names).distinct().tuples())
for query_res in query:
    target_names = query_res[0]
    if len(target_names) == 1:
        target_name = target_names[0]
    else:
        NotImplementedError('Multiple targets not implemented yet')
    print(target_name)
    parent_name = root_name + target_name + '/'
    subquery = (Network.select(Network.id, NetworkJSON.network_json)
                .where(Network.target_names == Param(target_names))
                .join(NetworkJSON)
                .tuples())
    for subquery_res in subquery:
        id, json_dict = subquery_res
        nn = QuaLiKizNDNN(json_dict)
        network_name = parent_name + str(id)
        if network_name in store:
            continue
        else:
            print('Generating ', network_name)
            df_nn = nn.get_output(**input)
            df_nn.index = input.index
            store[network_name] = df_nn
embed()
#df_nn = nn.get_outputs(**input)
#df_nn.index = input.index
#store[] = df_nn
