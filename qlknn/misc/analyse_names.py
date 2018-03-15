import re

from IPython import embed

particle_rotationless_diffusion_vars = [u'df', u'vt', u'vc']
particle_rotation_diffusion_vars = [u'vr']
particle_diffusion_vars = particle_rotationless_diffusion_vars + particle_rotation_diffusion_vars
particle_flux = [u'pf']
particle_vars = particle_flux + particle_diffusion_vars
heat_flux = [u'ef']
heat_vars = heat_flux
momentum_flux = [u'vf']
momentum_vars = momentum_flux
rotation_vars = particle_rotation_diffusion_vars

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
            flux &= split_name(name)[0] in heat_flux + particle_flux + momentum_flux
    except ValueError:
        flux = False
    return flux

def is_transport(name):
    transport = True
    try:
        for part_name in split_parts(name):
            transport &= split_name(name)[0] in heat_vars + particle_vars + momentum_vars
    except ValueError:
        transport = False
    return transport

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

def contains_sep(name):
    return any(sub in name for sub in ['TEM', 'ITG', 'ETG'])

def is_full_transport(name):
    return is_transport(name) and not contains_sep(name)

def is_transport_family(name, identifiers, combiner):
    if is_transport(name):
        subnames = split_name(name)
        transport_family = any(sub in subnames[0] for sub in identifiers)
        for subname in subnames[1:]:
            transport_family = combiner(transport_family, any(sub in subname for sub in identifiers))
    else:
        transport_family = False
    return transport_family

def is_pure_diffusion(name):
    return is_transport_family(name, particle_diffusion_vars, lambda x, y: x and y)

def is_pure_heat(name):
    return is_transport_family(name, heat_vars, lambda x, y: x and y)

def is_pure_particle(name):
    return is_transport_family(name, particle_vars, lambda x, y: x and y)

def is_pure_rot(name):
    return is_transport_family(name, rotation_vars, lambda x, y: x and y)

def is_partial_diffusion(name):
    return is_transport_family(name, particle_diffusion_vars, lambda x, y: x or y)

def is_partial_heat(name):
    return is_transport_family(name, heat_vars, lambda x, y: x or y)

def is_partial_particle(name):
    return is_transport_family(name, particle_vars, lambda x, y: x or y)

def is_partial_rot(name):
    return is_transport_family(name, rotation_vars, lambda x, y: x or y)

def is_leading(name):
    if is_transport(name):
        leading = True
        if not is_full_transport(name):
            if any(sub in name for sub in ['div', 'plus']):
                if 'ITG' in name:
                    if name not in ['efeITG_GB_div_efiITG_GB', 'pfeITG_GB_div_efiITG_GB']:
                        leading = False
                elif 'TEM' in name:
                    if name not in ['efiTEM_GB_div_efeTEM_GB', 'pfeTEM_GB_div_efeTEM_GB']:
                        leading = False
            if 'pfi' in name:
                leading = False
    else:
        leading = False
    return leading

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
