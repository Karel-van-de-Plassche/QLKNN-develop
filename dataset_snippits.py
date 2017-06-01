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

# sepflux based filter
        sepflux = sepflux.loc[index]
        for flux in ['efeETG_GB',
                     'efeITG_GB',
                     'efeTEM_GB',
                     'efiITG_GB',
                     'efiTEM_GB']:

            index = sepflux.index[(sepflux[flux] > min) & (sepflux[flux] < max)]
