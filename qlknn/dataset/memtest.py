from data_io import *
import sys
from IPython import embed

dim = 7
gen = 3
filter_num = 8

root_dir = '../../'
basename = ''.join(['gen', str(gen), '_', str(dim), 'D_nions0_flat', '_filter', str(filter_num)])
store_name = root_dir + basename + '.h5.1'

if len(sys.argv) > 1: #Ran from command line
    suffix = sys.argv[1]
    fast = sys.argv[2] == 'True'
    mode = sys.argv[3]
    columns = sys.argv[4:]
    if columns == []:
        columns = None
    elif columns == ['False']:
        columns = False
    print(columns)
    load_from_store(store_name + suffix, fast, mode, columns=columns)
else:#Ran directly, run timing test!
    suffix = '_sep'
    import time
    for columns in [['efeITG_GB', 'efiITG_GB', 'pfeITG_GB'], None]:
        print('For columns', columns)
        for mode in ['old', 'fast', 'bare', 'join', 'concat', 'merge', 'assign']:
            #update too slow..
            starttime = time.time()
            if mode == 'old':
                load_from_store(store_name, columns=columns)
            elif mode == 'fast':
                load_from_store(store_name + suffix, fast=True, columns=columns)
            else:
                load_from_store(store_name + suffix, fast=False, mode=mode, columns=columns)
            print('{!s:<6} took {:.1f}s'.format(mode, time.time() - starttime))
