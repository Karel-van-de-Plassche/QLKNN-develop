from IPython import embed
import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0,'..')

from run_model import QuaLiKizDuoNN, QuaLiKizMultiNN, QuaLiKizNDNN, QuaLiKizComboNN

simple_nns = ['efe_GB',
              'efi_GB',
              ]
#              'dfe_GB',
#              'dfi_GB',
#              'gam_GB_less2max',
#              'gam_GB_leq2max',
#              'vte_GB_plus_vce_GB',
#              'vti_GB_plus_vci_GB']
#
#combo_nns = {'efe_GB2': ['efe_GB_min_efeETG_GB', 'efeETG_GB', lambda x, y: x + y],
#             'vte_GB_plus_vce_GB2': ['vte_GB', 'vce_GB', lambda x, y: x + y],
#             'vti_GB_plus_vci_GB2': ['vti_GB', 'vci_GB', lambda x, y: x + y]
#}

#nns = []
#for name in simple_nns:
#    nn = QuaLiKizNDNN.from_json('nns/nn_' + name + '.json')
#    nns.append(nn)
nn_dict = {}
path = 'nns'
for file_ in os.listdir(path):
    if file_.endswith('.json'):
        nn_dict[file_[3:-5]] = QuaLiKizNDNN.from_json(os.path.join(path, file_))
#efe_fancy = 1. + (3.) / (2. + 1) + (5.) / (4. + 1)
#efi_fancy = (2. * 3.) / (2. + 1) + (4. * 5.) / (4. + 1)

efe_GB_A = QuaLiKizComboNN('efe_GB_A', [nn_dict['efeETG_GB'],
                                        nn_dict['efiITG_GB_div_efeITG_GB'],
                                        nn_dict['efiITG_GB_plus_efeITG_GB'],
                                        nn_dict['efiTEM_GB_div_efeTEM_GB'],
                                        nn_dict['efiTEM_GB_plus_efeTEM_GB']],
                            lambda a, b, c, d, e: a + c / (b + 1) + e / (d + 1))
efi_GB_A = QuaLiKizComboNN('efi_GB_A', [
                                        nn_dict['efiITG_GB_div_efeITG_GB'],
                                        nn_dict['efiITG_GB_plus_efeITG_GB'],
                                        nn_dict['efiTEM_GB_div_efeTEM_GB'],
                                        nn_dict['efiTEM_GB_plus_efeTEM_GB']],
                            lambda b, c, d, e: (b * c) / (b + 1) + (d * e) / (d + 1))
efe_GB_C = QuaLiKizComboNN('efe_GB_C', [nn_dict['efi_GB_div_efe_GB'],
                                        nn_dict['efi_GB_plus_efe_GB']],
                           lambda a, b: b / (a + 1))
efi_GB_C = QuaLiKizComboNN('efi_GB_C', [nn_dict['efi_GB_div_efe_GB'],
                                        nn_dict['efi_GB_plus_efe_GB']],
                           lambda a, b: (a * b) / (a + 1))
efe_GB_D = nn_dict['efe_GB']
efi_GB_D = nn_dict['efi_GB']
nns = [
    efe_GB_A,
    efi_GB_A,
    efe_GB_C,
    efi_GB_C,
    efe_GB_D,
    efi_GB_D,
    nn_dict['efeTEM_GB'],
    nn_dict['efeETG_GB'],
    nn_dict['efeITG_GB'],
    nn_dict['efiTEM_GB'],
    nn_dict['efiITG_GB']
]
nn = QuaLiKizMultiNN(nns)
#
#for name, recipe in combo_nns.items():
#    nn1 = QuaLiKizNDNN.from_json('nns/nn_' + recipe[0] + '.json')
#    nn2 = QuaLiKizNDNN.from_json('nns/nn_' + recipe[1] + '.json')
#    nn = QuaLiKizDuoNN(name, nn1, nn2, recipe[2])
#    nns.append(nn)

#nn = QuaLiKizMultiNN(nns)
if __name__ == '__main__':
    scann = 24
    input = pd.DataFrame()
    input['Ati'] = np.array(np.linspace(2,13, scann))
    input['Ti_Te']  = np.full_like(input['Ati'], 1.)
    #input['Zeffx']  = np.full_like(input['Ati'], 1.)
    input['An']  = np.full_like(input['Ati'], 2.)
    input['Ate']  = np.full_like(input['Ati'], 5.)
    input['qx'] = np.full_like(input['Ati'], 0.660156)
    input['smag']  = np.full_like(input['Ati'], 0.399902)
    input['Nustar']  = np.full_like(input['Ati'], 0.009995)
    input['x']  = np.full_like(input['Ati'], 0.449951)
    fluxes = nn.get_outputs(**input)
    print(fluxes)
    embed()
