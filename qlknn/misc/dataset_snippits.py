# First Raptor 9D NNs
list_train_dims = ['efe_GB',
                   'efeETG_GB',
                   ['efe_GB', 'min', 'efeETG_GB'],
                   'efi_GB',
                   ['vte_GB', 'plus', 'vce_GB'],
                   ['vti_GB', 'plus', 'vci_GB'],
                   'dfe_GB',
                   'dfi_GB',
                   'vte_GB',
                   'vce_GB',
                   'vti_GB',
                   'vci_GB',
                   'gam_GB_less2max',
                   'gam_GB_leq2max']
# everything_nions0_zeffx1_nustar1e-3_sepfluxes.h5
list_train_dims = ['efe_GB',
                   'efi_GB',
                   'efiITG_GB',
                   'efiTEM_GB',
                   'efeETG_GB',
                   'efeITG_GB',
                   'efeTEM_GB',
                   'gam_GB_less2max',
                   'gam_GB_leq2max']

        index = input.index[(
                             np.isclose(input['Zeffx'], 1,     atol=1e-5, rtol=1e-3) &
                             np.isclose(input['Nustar'], 1e-3, atol=1e-5, rtol=1e-3)
                             )]

# filtered_clipped_nions0_zeffx1_nustar1e-3_sepfluxes_0_60.h5
list_train_dims = ['efe_GB',
                   'efi_GB',
                   'efiITG_GB',
                   'efiTEM_GB',
                   'efeETG_GB',
                   'efeITG_GB',
                   'efeTEM_GB',
                   ['efiITG_GB', 'div', 'efeITG_GB'],
                   ['efiITG_GB', 'plus', 'efeITG_GB'],
                   ['efiTEM_GB', 'div', 'efeTEM_GB'],
                   ['efiTEM_GB', 'plus', 'efeTEM_GB'],
                   'gam_GB_less2max',
                   'gam_GB_leq2max']
    max = 60
    min = 0.1
        index = input.index[(
                             np.isclose(input['Zeffx'], 1,     atol=1e-5, rtol=1e-3) &
                             np.isclose(input['Nustar'], 1e-3, atol=1e-5, rtol=1e-3)
                             )]
        sepflux = sepflux.loc[index]
        for flux in ['efeETG_GB',
                     'efeITG_GB',
                     'efeTEM_GB',
                     'efiITG_GB',
                     'efiTEM_GB']:

            sepflux = sepflux.loc[(sepflux[flux] >= min) & (sepflux[flux] < max)]
        index = sepflux.index
        totflux = totflux.loc[index]
        for flux in ['efe_GB',
                     'efi_GB']:
            totflux = totflux.loc[(totflux[flux] >= min) & (totflux[flux] < max)]
        index = totflux.index

        if 'gam' not in name and 'input' not in name and 'index' not in name:
            var = var.loc[var != 0]
        if 'efi' in name and 'efe' not in name:
            print('efi_style')
            var = var.loc[(gam_less != 0).iloc[:, 0]]
        elif 'efe' in name and 'efi' not in name:
            print('efe_style')
            var = var.loc[(gam_leq != 0).iloc[:, 0]]
        elif 'efe' in name and 'efi' in name:
            print('mixed_style')
            var = var.loc[(gam_less != 0).iloc[:, 0]]
            var = var.loc[(var != np.inf) & (var != -np.inf) & (var != np.nan)]
        elif 'index' in name:
            pass
        else:
            print('weird_style')
            pass

# sepflux based filter
    max = 60
    min = 0
        sepflux = sepflux.loc[index]
        for flux in ['efeETG_GB',
                     'efeITG_GB',
                     'efeTEM_GB',
                     'efiITG_GB',
                     'efiTEM_GB']:

            index = sepflux.index[(sepflux[flux] > min) & (sepflux[flux] < max)]
