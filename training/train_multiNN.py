import os
import subprocess as sp

nns_dir = os.path.abspath('nns')
for name in os.listdir(nns_dir):
    print(name)
    dir = os.path.join(nns_dir, name)
    os.chdir(dir)
    print('start training')
    result = sp.run(['python3', 'train_NDNN.py'], stdout=sp.PIPE, stderr=sp.PIPE)
    print(result)
    print('training done')
