import re

from IPython import embed

particle_diffusion_vars = [u'df', u'vt', u'vr', u'vc']
particle_vars = [u'pf'] + particle_diffusion_vars
heat_vars = [u'ef']
momentum_vars = [u'vf']

def split_parts(name):
    splitted = re.compile('((?:.{2})(?:.)(?:|ITG|ETG|TEM)_(?:GB|SI|cm))').split(name)
    if splitted[0] != '' or splitted[-1] != '':
        raise ValueError('Split {!s} in an unexpected way: {!s}'.format(name, splitted))
    del splitted[0], splitted[-1]
    return splitted

def extract_part_names(splitted):
    return splitted[slice(0, len(splitted), 2)]

def extract_operations(splitted):
    return splitted[slice(1, len(splitted) - 1, 2)]

def is_pure(name):
    try:
        pure = len(split_parts(name)) == 1
    except ValueError:
        pure = False
    return pure

def is_flux(name):
    flux = True
    try:
        for part_name in split_parts(name):
            flux &= split_name(name)[0] in heat_vars + particle_vars + momentum_vars
    except ValueError:
        flux = False
    return flux

def is_pure_flux(name):
    try:
        flux = split_name(name)[0] in heat_vars + particle_vars + momentum_vars
        pure_flux = is_pure(name) and flux
    except ValueError:
        pure_flux = False
    return pure_flux

def split_name(name):
    splitted = re.compile('(.{2})(.)(|ITG|ETG|TEM)_(GB|SI|cm)').split(name)
    if splitted[0] != '' or splitted[-1] != '':
        raise ValueError('Split {!s} in an unexpected way: {!s}'.format(name, splitted))
    del splitted[0], splitted[-1]
    return splitted

def is_growth(name):
    return name in ['gam_leq_GB', 'gam_great_GB']

if __name__ == '__main__':
    print(split_parts('efeITG_GB_div_efiITG_GB_plus_pfeITG_GB'))
    print(split_parts('efeITG_GB_div_efiITG_GB'))
    print(split_parts('efeITG_GB'))

    print(extract_part_names(split_parts('efeITG_GB_div_efiITG_GB_plus_pfeITG_GB')))
    print(extract_part_names(split_parts('efeITG_GB_div_efiITG_GB')))
    print(extract_part_names(split_parts('efeITG_GB')))

    print(extract_operations(split_parts('efeITG_GB_div_efiITG_GB_plus_pfeITG_GB')))
    print(extract_operations(split_parts('efeITG_GB_div_efiITG_GB')))
    print(extract_operations(split_parts('efeITG_GB')))

    print(is_pure('efeITG_GB_div_efiITG_GB_plus_pfeITG_GB'))
    print(is_pure('efeITG_GB_div_efiITG_GB'))
    print(is_pure('efeITG_GB'))

    print(is_pure_flux('efeITG_GB_div_efiITG_GB_plus_pfeITG_GB'))
    print(is_pure_flux('efeITG_GB_div_efiITG_GB'))
    print(is_pure_flux('efeITG_GB'))

    print(is_flux('efeITG_GB_div_efiITG_GB_plus_pfeITG_GB'))
    print(is_flux('efeITG_GB_div_efiITG_GB'))
    print(is_flux('efeITG_GB'))
