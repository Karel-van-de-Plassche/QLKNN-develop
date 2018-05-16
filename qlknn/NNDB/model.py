import inspect
import os
import sys
import json
import subprocess
import socket
import re
import traceback
import operator
from warnings import warn
from functools import reduce
from itertools import chain
from collections import OrderedDict
import itertools
import copy

import numpy as np
import scipy.io as io
import pandas as pd
from peewee import (Model, prefetch,
                    FloatField, FloatField, IntegerField, BooleanField, TextField, ForeignKeyField, DateTimeField,
                    ProgrammingError, AsIs, fn, SQL, DoesNotExist, JOIN)
from playhouse.postgres_ext import PostgresqlExtDatabase, ArrayField, BinaryJSONField, JSONField, HStoreField
from playhouse.hybrid import hybrid_property
#from playhouse.shortcuts import RetryOperationalError #peewee==2.10.1
from IPython import embed

from qlknn.models.ffnn import QuaLiKizNDNN, QuaLiKizComboNN, nn_dict_to_matlab
from qlknn.misc.analyse_names import split_name

#class RetryPostgresqlExtDatabase(RetryOperationalError, PostgresqlExtDatabase):
#    pass
#db = RetryPostgresqlExtDatabase(database='nndb', host='gkdb.org')
db = PostgresqlExtDatabase(database='nndb', host='gkdb.org', register_hstore=True)
#db.execute_sql('CREATE SCHEMA IF NOT EXISTS develop;')
def flatten(l):
    return [item for sublist in l for item in sublist]

class BaseModel(Model):
    """A base model that will use our Postgresql database"""
    class Meta:
        database = db
        schema = 'develop'

class TrainScript(BaseModel):
    script = TextField()
    version = TextField()

    @classmethod
    @db.atomic()
    def from_file(cls, pwd):
        with open(pwd, 'r') as script:
            script = script.read()

        train_script_query = TrainScript.select().where(TrainScript.script == script)
        if train_script_query.count() == 0:
            stdout = subprocess.check_output('git rev-parse HEAD',
                                       shell=True)
            version = stdout.decode('UTF-8').strip()
            train_script = TrainScript(
                script=script,
                version=version
            )
            train_script.save()
        elif train_script_query.count() == 1:
            train_script = train_script_query.get()
        else:
            raise Exception('multiple train scripts found. Could not choose')
        return train_script

class Filter(BaseModel):
    script = TextField()
    hypercube_script = TextField(null=True)
    description = TextField(null=True)
    min = FloatField(null=True)
    max = FloatField(null=True)
    remove_negative = BooleanField(null=True)
    remove_zeros = BooleanField(null=True)
    gam_filter = BooleanField(null=True)
    ck_max = FloatField(null=True)
    diffsep_max = FloatField(null=True)

    @classmethod
    @db.atomic()
    def from_file(cls, filter_file, hyper_file):
        with open(filter_file, 'r') as script:
            filter_script = script.read()

        with open(hyper_file, 'r') as script:
            hypercube_script = script.read()

        filter = Filter(script=filter_script, hypercube_script=hypercube_script)
        filter.save()

    @classmethod
    def find_by_path_name(cls, name):
        split = re.split('(?:(unstable)_|)(sane|test|training)_(?:gen(\d+)_|)(\d+)D_nions0_flat_filter(\d+).h5', name)
        try:
            if len(split) != 7:
                raise Exception
            filter_id = int(split[5])
        except:
            raise Exception('Could not find filter ID from name "{!s}"'.format(name))
        return filter_id


class Network(BaseModel):
    feature_names = ArrayField(TextField)
    target_names = ArrayField(TextField)
    recipe = TextField(null=True)
    networks = ArrayField(IntegerField, null=True)


    @classmethod
    def get_recursive_subquery(cls, params, table=None, distinct=True):
        if table is None:
            table = Hyperparameters
        if not isinstance(params, list):
            params = [params]
        from peewee import CTE, Table, Select
        #params = ['cost_l2_scale', 'hidden_neurons']
        recursetree = Table('recursetree')
        non_rec = (Network.select(Network.id, Network.networks, Network.id.alias('root'))
                   #.where(Network.recipe == 'np.hstack(args)')
                   .where((Network.networks.is_null(False)))
                   )
        rec = (Network.select(Network.id, Network.networks, recursetree.c.root)
               .from_(Network, recursetree)
               .where(Network.id == fn.ANY(recursetree.c.pure_children)))
        cte = (non_rec + rec).cte('recursetree', recursive=True, columns=['net_id', 'pure_children', 'root'])
        subquery_params = [getattr(table, param) for param in params]

        subquery = (Select(columns=[cte.c.net_id, cte.c.root] + subquery_params)
                 .from_(cte)
                 .bind(db)
                 .join(PureNetworkParams, on=(PureNetworkParams.network == cte.c.net_id))
                 .join(table,   on=(table.pure_network_params == PureNetworkParams.id))
                 .group_by([cte.c.net_id, cte.c.root] + subquery_params)
                 .where((cte.c.pure_children.is_null(True)))
                 ).alias('childq')

        query_params = [getattr(subquery.c, param) for param in params]
        if distinct:
            query_params = [qp.distinct() for qp in query_params]
        query_params = [fn.ARRAY_AGG(qp, coerce=False).alias(param) for param, qp in zip(params, query_params)]
        root = fn.int8(subquery.c.root).alias('root')
        pure_children = fn.ARRAY_AGG(subquery.c.net_id).alias('pure_children')
        #*[fn.ARRAY_AGG(getattr(cls, name)) for name in cls._meta.fields.keys()], 
        netself = cls.alias('net')
        query = (cls.select(root, pure_children, netself, *query_params)
                 .from_(subquery)
                 .group_by(subquery.c.root, *[getattr(netself, name) for name in netself._meta.fields.keys()])
                 .join(cls, on=(subquery.c.net_id == cls.id))
                 .join(netself, on=(subquery.c.root == netself.id))
                 .with_cte(cte)
                 )
                 #.join(subquery, on=(Network.id == subquery.c.net_id))
        return query

    @classmethod
    def get_double_subquery(cls, params, table=None):
        if table is None:
            table = Hyperparameters
        if not isinstance(params, list):
            params = [params]
        x = Network.alias()
        y = Network.alias()
        subx = x.select(x.id, x.networks)
        suby = y.select(y.id)

        subq = (cls.select(Network.id,
                  *[fn.ARRAY_AGG(getattr(table, param), coerce=False).alias(param)
                   for param in params])
         .join(subx, JOIN.LEFT_OUTER, on=subx.c.id == fn.ANY(Network.networks))
         .join(suby, JOIN.LEFT_OUTER, on=suby.c.id == fn.ANY(subx.c.networks))
         .join(PureNetworkParams, on=((PureNetworkParams.network_id == Network.id) |
                                   (PureNetworkParams.network_id == subx.c.id) |
                                   (PureNetworkParams.network_id == suby.c.id)))
         .join(table, on=PureNetworkParams.id == table.pure_network_params_id)
         .group_by(Network.id)
        )
        return subq

    @classmethod
    def find_divsum_candidates(cls):
        query = (cls
                 .select()
                 .where(cls.target_names[0] % '%_div_%')
                 .where(fn.array_length(cls.target_names, 1) == 1)
                 )
        for pure_network_params in query:
            try:
                cls.divsum_from_div_id(pure_network_params.id)
            except NotImplementedError:
                formatted_lines = traceback.format_exc().splitlines()
                print(formatted_lines[-1])
            except Exception:
                traceback.print_exc()
                raise

    @staticmethod
    def generate_divsum_recipes(target_name):
        splitted = re.compile('(.*)_(div|plus)_(.*)').split(target_name)
        if len(splitted) != 5:
            raise ValueError('Could not split {!s} in divsum parts'.format(target_name))

        partner_target_sets = []
        formula_sets = []
        if splitted[2] == 'div':
            if splitted[1].startswith('efi') and splitted[3].startswith('efe'):
                # If it is efi / efe
                # Old style: efi / efe == nn0, efi + efe == nn1
                #partner_targets = [[splitted[1] + '_plus_' + splitted[3]]]
                #formulas = OrderedDict([(splitted[1], '(nn{0:d} * nn{1:d}) / (nn{0:d} + 1)'),
                #                        (splitted[3], '(nn{1:d}) / (nn{0:d} + 1)')])
                #partner_target_sets.append(partner_targets)
                #formula_sets.append(formulas)
                # Simple style: efi / efe == nn0, efe == nn1
                efe = splitted[3]
                efi = splitted[1]
                partner_targets = [[efe]]
                formulas = OrderedDict([
                    (efe, 'nn{1:d}'),
                    (efi, '(nn{0:d} * nn{1:d})')
                ])
                partner_target_sets.append(partner_targets)
                formula_sets.append(formulas)
            elif splitted[1].startswith('efe') and splitted[3].startswith('efi'):
                # If it is efe / efi
                # Simple style: efe / efi == nn0, efi == nn1
                efe = splitted[1]
                efi = splitted[3]
                partner_targets = [[efi]]
                formulas = OrderedDict([
                    (efe, '(nn{0:d} * nn{1:d})'),
                    (efi, 'nn{1:d}')
                ])
                partner_target_sets.append(partner_targets)
                formula_sets.append(formulas)
            elif splitted[1].startswith('pfe') and splitted[3].startswith('efi'):
                # If it is pfe / efi
                pfe = splitted[1]
                efi = splitted[3]
                split_efi = re.compile('(?=.*)(.)(|ITG|ETG|TEM)(_GB|SI|cm)').split(efi)
                efe = ''.join(*[[split_efi[0]] + ['e'] + split_efi[2:]])
                ## Triplet style: pfe / efi == nn0, pfe + efi + efe == nn1, efi / efe == nn2
                #partner_targets = [[pfe + '_plus_' + efi + '_plus_' + efe],
                #                   [efi + '_div_' + efe]
                #                   ]
                #formulas = OrderedDict([
                #    (efi, '(nn{1:d} * nn{2:d}) / (1 + nn{0:d} + nn{2:d})'),
                #    (efe, 'nn{1:d} / (1 + nn{0:d} + nn{2:d})'),
                #    (pfe, '(nn{0:d} * nn{1:d}) / (1 + nn{0:d} + nn{2:d})')
                #])
                #partner_target_sets.append(partner_targets)
                #formula_sets.append(formulas)
                # Simple style: pfe / efi == nn0, efi == nn1, efe / efi == nn2
                partner_targets = [[efi],
                                   [efe + '_div_' + efi]
                                   ]
                formulas = OrderedDict([
                    (efe, '(nn{2:d} * nn{1:d})'),
                    (efi, 'nn{1:d}'),
                    (pfe, '(nn{0:d} * nn{1:d})')
                ])
                partner_target_sets.append(partner_targets)
                formula_sets.append(formulas)
            elif splitted[1].startswith('efi') and splitted[3].startswith('pfe'):
                raise NotImplementedError('Should look at those again..')
                # If it is efi / pfe
                efi = splitted[1]
                pfe = splitted[3]
                split_efi = re.compile('(?=.*)(.)(|ITG|ETG|TEM)(_GB|SI|cm)').split(efi)
                efe = ''.join(*[[split_efi[0]] + ['e'] + split_efi[2:]])
                ## Triplet style: efi / pfe == nn0, pfe + efi + efe == nn1, efi / efe == nn2
                #partner_targets = [[pfe + '_plus_' + efi + '_plus_' + efe],
                #                   [efi + '_div_' + efe]
                #                   ]
                #formulas = OrderedDict([
                #    (efi, '(nn{0:d} * nn{1:d} * nn{2:d}) / (nn{0:d} + nn{2:d} + nn{0:d} * nn{2:d})'),
                #    (efe, '(nn{0:d} * nn{1:d}) / (nn{0:d} + nn{2:d} + nn{0:d} * nn{2:d})'),
                #    (pfe, '(nn{1:d} * nn{2:d}) / (nn{0:d} + nn{2:d} + nn{0:d} * nn{2:d})')
                #])
                #partner_target_sets.append(partner_targets)
                #formula_sets.append(formulas)
                ## Heatflux style: efi / pfe == nn0, efi + efe == nn1, efi / efe == nn2
                #partner_targets = [[efi + '_plus_' + efe],
                #                   [efi + '_div_' + efe]
                #                   ]
                #formulas = OrderedDict([
                #    (efi, '(nn{1:d} * nn{2:d}) / (1 + nn{2:d})'),
                #    (efe, '(nn{1:d}) / (1 + nn{2:d})'),
                #    (pfe, '(nn{1:d} * nn{2:d}) / (nn{0:d} * (1 + nn{2:d}))')
                #])
                #partner_target_sets.append(partner_targets)
                #formula_sets.append(formulas)
            elif splitted[1].startswith('pfe') and splitted[3].startswith('efe'):
                # If it is pfe / efe
                pfe = splitted[1]
                efe = splitted[3]
                split_efe = re.compile('(?=.*)(.)(|ITG|ETG|TEM)(_GB|SI|cm)').split(efe)
                efi = ''.join(*[[split_efe[0]] + ['i'] + split_efe[2:]])
                ## Triplet style: pfe / efe == nn0, pfe + efi + efe == nn1, efi / efe == nn2
                #partner_targets = [[pfe + '_plus_' + efi + '_plus_' + efe],
                #                   [efi + '_div_' + efe]
                #                   ]
                #formulas = OrderedDict([
                #    (efi, '(nn{1:d} * nn{2:d}) / (1 + nn{0:d} + nn{2:d})'),
                #    (efe, '(nn{1:d}) / (1 + nn{1:d} + nn{2:d})'),
                #    (pfe, '(nn{0:d} * nn{1:d}) / (1 + nn{0:d} + nn{2:d})')
                #])
                #partner_target_sets.append(partner_targets)
                #formula_sets.append(formulas)
                ## Heatflux style: pfe / efe == nn0, efi + efe == nn1, efi / efe == nn2
                #partner_targets = [[efi + '_plus_' + efe],
                #                   [efi + '_div_' + efe]
                #                   ]
                #formulas = OrderedDict([
                #    (efi, '(nn{1:d} * nn{2:d}) / (1 + nn{2:d})'),
                #    (efe, '(nn{1:d}) / (1 + nn{2:d})'),
                #    (pfe, '(nn{0:d} * nn{1:d} * nn{2:d}) / (1 + nn{2:d})')
                #])
                #partner_target_sets.append(partner_targets)
                #formula_sets.append(formulas)
                # Simple style: pfe/efe == nn0, efe == nn1, efi / efe == nn2
                partner_targets = [[efe],
                                   [efi + '_div_' + efe]
                                   ]
                formulas = OrderedDict([
                    (efe, 'nn{1:d}'),
                    (efi, '(nn{2:d} * nn{1:d})'),
                    (pfe, '(nn{0:d} * nn{1:d})')
                ])
                partner_target_sets.append(partner_targets)
                formula_sets.append(formulas)
            elif splitted[1].startswith('dfe') and splitted[3].startswith('efe'):
                # If it is dfe / efe
                efe = splitted[3]
                transp, species, mode, norm = split_name(efe)
                efi = ''.join(['efi', mode, '_', norm])
                dfe = splitted[1]
                dfi = ''.join(['dfi', mode, '_', norm])
                vte = ''.join(['vte', mode, '_', norm])
                vti = ''.join(['vti', mode, '_', norm])
                vce = ''.join(['vce', mode, '_', norm])
                vci = ''.join(['vci', mode, '_', norm])
                # Simple style: dfe/efe == nn0, efe == nn1, efi / efe == nn2, dfi / efe == nn3, vte/efe == nn4, vti/efe == nn5, vce/efe == nn6, vci/efe == nn7
                partner_targets = [[efe],
                                   [efi + '_div_' + efe],
                                   [dfi + '_div_' + efe],
                                   [vte + '_div_' + efe],
                                   [vti + '_div_' + efe],
                                   [vce + '_div_' + efe],
                                   [vci + '_div_' + efe]
                                   ]
                formulas = OrderedDict([
                    (efe, 'nn{1:d}'),
                    (efi, '(nn{2:d} * nn{1:d})'),
                    (dfe, '(nn{0:d} * nn{1:d})'),
                    (dfi, '(nn{3:d} * nn{1:d})'),
                    (vte, '(nn{4:d} * nn{1:d})'),
                    (vti, '(nn{5:d} * nn{1:d})'),
                    (vce, '(nn{6:d} * nn{1:d})'),
                    (vci, '(nn{7:d} * nn{1:d})')
                ])
                partner_target_sets.append(partner_targets)
                formula_sets.append(formulas)
            elif splitted[1].startswith('dfe') and splitted[3].startswith('efi'):
                # If it is dfe / efe
                efi = splitted[3]
                transp, species, mode, norm = split_name(efi)
                efe = ''.join(['efe', mode, '_', norm])
                dfe = splitted[1]
                dfi = ''.join(['dfi', mode, '_', norm])
                vte = ''.join(['vte', mode, '_', norm])
                vti = ''.join(['vti', mode, '_', norm])
                vce = ''.join(['vce', mode, '_', norm])
                vci = ''.join(['vci', mode, '_', norm])
                # Simple style: dfe/efi == nn0, efi == nn1, efe / efi == nn2, dfi / efi == nn3, vte/efi == nn4, vti/efi == nn5, vce/efi == nn6, vci/efi == nn7
                partner_targets = [[efi],
                                   [efe + '_div_' + efi],
                                   [dfi + '_div_' + efi],
                                   [vte + '_div_' + efi],
                                   [vti + '_div_' + efi],
                                   [vce + '_div_' + efi],
                                   [vci + '_div_' + efi]
                                   ]
                formulas = OrderedDict([
                    (efe, '(nn{2:d} * nn{1:d})'),
                    (efi, 'nn{1:d}'),
                    (dfe, '(nn{0:d} * nn{1:d})'),
                    (dfi, '(nn{3:d} * nn{1:d})'),
                    (vte, '(nn{4:d} * nn{1:d})'),
                    (vti, '(nn{5:d} * nn{1:d})'),
                    (vce, '(nn{6:d} * nn{1:d})'),
                    (vci, '(nn{7:d} * nn{1:d})')
                ])
                partner_target_sets.append(partner_targets)
                formula_sets.append(formulas)
            else:
                raise NotImplementedError("Div style network with target {!s} and first part '{!s}'".format(target_name, splitted[1]))
        else:
            raise Exception('Divsum network needs div network, not {!s}'.format(target_name))
        return formula_sets, partner_target_sets

    @classmethod
    def mixed_nets_from_id(cls, network_id, raise_on_missing=False):
        nn = cls.get_by_id(network_id)
        if len(nn.target_names) != 1:
            raise ValueError('Mixed network generation only defined for single target,' +
                             'not {!s}'.format(nn.target_names))
        target_name = nn.target_names[0]

        trigger_name = 'vciTEM_GB_div_efeTEM_GB'
        if target_name != trigger_name:
            raise ValueError("Target name '{!s}' != trigger name '{!s}'".format(target_name, trigger_name))

        print('Trying to create mega-mixed Network {:d} with target_name {!s}'.format(nn.id, target_name))
        from qlknn.misc.analyse_names import split_name, split_parts
        #splitted = split_parts(target_name)
        #if len(splitted) != 3 or splitted[1] != 'div':
        #    raise ValueError('Could not split {!s} in div parts'.format(target_name))
        #transp, species, mode, norm = split_name(target_name)

        partner_targets = [
            ['efeETG_GB'], #1
            ['efeITG_GB_div_efiITG_GB'],# 2
            ['efeTEM_GB'], #3
            ['efiITG_GB'], #4
            ['efiTEM_GB_div_efeTEM_GB'],# 5
            ['pfeITG_GB_div_efiITG_GB'],# 6
            ['pfeTEM_GB_div_efeTEM_GB'],# 7
            ['dfeITG_GB_div_efiITG_GB'],# 8
            ['dfeTEM_GB_div_efeTEM_GB'],# 9
            ['vteITG_GB_div_efiITG_GB'],# 10
            ['vteTEM_GB_div_efeTEM_GB'],# 11
            ['vceITG_GB_div_efiITG_GB'],# 12
            ['vceTEM_GB_div_efeTEM_GB'],# 13
            ['dfiITG_GB_div_efiITG_GB'],# 14
            ['dfiTEM_GB_div_efeTEM_GB'],# 15
            ['vtiITG_GB_div_efiITG_GB'],# 16
            ['vtiTEM_GB_div_efeTEM_GB'],# 17
            ['vciITG_GB_div_efiITG_GB'],# 18
            #['vciTEM_GB_div_efeTEM_GB']# 19
        ]

        partner_targets_todo = copy.deepcopy(partner_targets)

        # These are the pre-defined 'best' networks
        cost_l2_scales = {'efeETG_GB': 5e-5,
                          'efiITG_GB': 5e-5,
                          'efeTEM_GB': 5e-5}

        # Find the three leading flux networks
        unordered_partners = {}
        for name, val in cost_l2_scales.items():
            query = nn.find_pure_partners([name],
                                      ignore_networkpars=['cost_l2_scale'])
            query = query.where(Hyperparameters.cost_l2_scale == val)
            query &= (PureNetworkParams.select()
                      .where(Hyperparameters.cost_l2_scale.cast('numeric') ==
                             val)
                      .join(Hyperparameters)
                     )
            unordered_partners[name] = \
                    cls.find_single_partner_from_query(query,
                                                       partner_target=name)
            partner_targets_todo.remove([name])

        formulas = OrderedDict([
            ('efeETG_GB', 'nn{0:d}'), #1
            ('efeITG_GB', '(nn{1:d} * nn{3:d})'),# 2
            ('efeTEM_GB', 'nn{2:d}'), #3
            ('efiITG_GB', 'nn{3:d}'), #4
            ('efiTEM_GB', '(nn{4:d} * nn{2:d})'),# 5
            ('pfeITG_GB', '(nn{5:d} * nn{3:d})'),# 6
            ('pfeTEM_GB', '(nn{6:d} * nn{2:d})'),# 7
            ('dfeITG_GB', '(nn{7:d} * nn{3:d})'),# 8
            ('dfeTEM_GB', '(nn{8:d} * nn{2:d})'),# 9
            ('vteITG_GB', '(nn{9:d} * nn{3:d})'),# 10
            ('vteTEM_GB', '(nn{10:d} * nn{2:d})'),# 11
            ('vceITG_GB', '(nn{11:d} * nn{3:d})'),# 12
            ('vceTEM_GB', '(nn{12:d} * nn{2:d})'),# 13
            ('dfiITG_GB', '(nn{13:d} * nn{3:d})'),# 14
            ('dfiTEM_GB', '(nn{14:d} * nn{2:d})'),# 15
            ('vtiITG_GB', '(nn{15:d} * nn{3:d})'),# 16
            ('vtiTEM_GB', '(nn{16:d} * nn{2:d})'),# 17
            ('vciITG_GB', '(nn{17:d} * nn{3:d})'),# 18
            ('vciTEM_GB', '(nn{18:d} * nn{2:d})'),#19
        ])

        # Find the divsum networks that match the trigger network (nn)
        skip_multinet = False
        for partner_target in partner_targets_todo:
            if len(partner_target) > 1:
                raise NotImplementedError('Multi-D target networks')
            query = nn.find_pure_partners(partner_target)
            partner_net = nn.find_single_partner_from_query(query, partner_target, raise_on_missing=raise_on_missing)
            unordered_partners[partner_target[0]] = partner_net
            if partner_net is None:
                skip_multinet = True

        # Order matters here! Our trigger network is the last one
        network_ids = [unordered_partners[target[0]].id for target in partner_targets]
        network_ids.append(nn.id)
        feature_names = nn.feature_names

        # Now create all networks
        multinet = cls.create_combinets_from_formulas(feature_names, formulas, network_ids, skip_multinet=skip_multinet)

    def find_pure_partners(self, partner_target, ignore_networkpars=None):
        pure_network_params_id = self.pure_network_params.get().id
        q1 = PureNetworkParams.find_similar_topology_by_id(
            pure_network_params_id,
            match_train_dim=False)
        q2 = PureNetworkParams.find_similar_networkpar_by_id(
            pure_network_params_id,
            ignore_pars=ignore_networkpars,
            match_train_dim=False)
        q3 = (PureNetworkParams
             .select()
             .where(self.__class__.target_names == partner_target)
             .where(self.__class__.feature_names == self.feature_names)
             .join(self.__class__)
             )
        return q1 & q2 & q3

    @staticmethod
    def pick_candidate(query):
        cls = query.model
        try:
            if cls == Network:
                candidates = [(el.postprocess.get().rms, el.id) for el in query]
            elif cls == PureNetworkParams:
                candidates = [(el.network.postprocess.get().rms, el.id) for el in query]
        except Postprocess.DoesNotExist as ee:
            net_id = re.search('Params: \[(.*)\]', ee.args[0])[1]
            table_field = re.search('WHERE \("t1"."(.*)"', ee.args[0])[1]
            raise Exception('{!s} {!s} does not exist! Run postprocess.py'.format(table_field, net_id))
        sort = []
        for rms, cls_id in candidates:
            assert len(rms) == 1
            sort.append([rms[0], cls_id])
        sort = sorted(sort)
        print('Selected {1:d} with RMS val {0:.2f}'.format(*sort[0]))
        query = (cls
                 .select()
                 .where(cls.id == sort[0][1])
        )
        return query

    @classmethod
    def find_single_partner_from_query(cls, query, partner_target=None, raise_on_missing=False):
        if partner_target is None:
            partner_target = query.sql()

        if query.count() > 1:
            print('Found {:d} matches for {!s}'.format(query.count(), partner_target))
            query = cls.pick_candidate(query)
        elif query.count() == 0:
            if raise_on_missing:
                raise DoesNotExist('No {!s} with target {!s}!'.format(cls, partner_target))
            print('No {!s} with target {!s}! Skip MultiNet creation..'.format(cls, partner_target))
            partner_net = None

        if query.count() == 1:
            purenet = query.get()
            partner_net = purenet.network
        return partner_net

    @classmethod
    def divsum_from_div_id(cls, network_id, raise_on_missing=False):
        nn = cls.get_by_id(network_id)
        if len(nn.target_names) != 1:
            raise ValueError('Divsum network needs div network, not {!s}'.format(nn.target_names))
        target_name = nn.target_names[0]
        print()
        print('Trying to combine Network {:d} with target {!s}'.format(nn.id, target_name))
        formula_sets, partner_target_sets = cls.generate_divsum_recipes(target_name)

        for formulas, partner_targets in zip(formula_sets, partner_target_sets):
            nns = [nn]
            skip_multinet = False
            for partner_target in partner_targets:

                query = nn.find_pure_partners(partner_target)
                partner_net = nn.find_single_partner_from_query(query, partner_target=partner_target, raise_on_missing=raise_on_missing)
                nns.append(partner_net)
                if partner_net is None:
                    skip_multinet = True

            network_ids = [nn.id if nn is not None else None for nn in nns]
            feature_names = nn.feature_names

            if not None in network_ids:
                #Skip everything if one network could not be found
                # TODO: decide if we allow for creation of 'partial' networks
                multinet = cls.create_combinets_from_formulas(feature_names, formulas, network_ids, skip_multinet=skip_multinet)

    @classmethod
    def create_combinets_from_formulas(cls, feature_names, formulas, network_ids, skip_multinet=False):
        recipes = OrderedDict()
        for target, formula in formulas.items():
            recipes[target] = formula.format(*network_ids)
            #recipes[target] = formula.format(*list(range(len(network_ids))))

        nets = []
        for target, recipe in recipes.items():
            is_pure = lambda recipe: all([el not in recipe for el in ['+', '-', '/', '*']])
            if is_pure(recipe):
                net_id = int(recipe.replace('nn', ''))
                #net_id = network_ids[net_num]
                nets.append(Network.get_by_id(net_id))
            else:
                net_idx = [int(id) for id in re.findall('nn(\d*)', recipe)]
                new_recipe = recipe
                for ii, net_id in enumerate(net_idx):
                    new_recipe = new_recipe.replace('nn' + str(net_id), 'nn' + str(ii))
                #child_nets = [network_ids[id] for id in net_idx]
                query = (Network.select()
                         .where((Network.recipe == new_recipe) &
                                (Network.networks == net_idx))
                         )
                if query.count() == 0:
                    combonet = cls(target_names=[target],
                                   feature_names=feature_names,
                                   recipe=new_recipe,
                                   networks=net_idx)
                    #raise Exception(combonet.recipe + ' ' + str(combonet.networks))
                    combonet.save()
                    print('Created ComboNetwork {:d} with recipe {!s} and networks {!s}'.format(combonet.id, recipe, network_ids))
                elif query.count() == 1:
                    combonet = query.get()
                    print('Network with recipe {!s} and networks {!s} already exists with id {!s}! Skip combonet creation!'.format(recipe, network_ids, combonet.id))
                else:
                    print('Insanity! Duplicate recipes! How could this happen..?')
                    embed()

                nets.append(combonet)

        multinet = None
        if not skip_multinet:
            try:
                multinet = cls.get(
                    cls.recipe == 'np.hstack(args)',
                    cls.networks == [net.id for net in nets],
                    cls.target_names == list(recipes.keys()),
                    cls.feature_names == feature_names
                )
            except Network.DoesNotExist:
                multinet = cls.create(
                    recipe = 'np.hstack(args)',
                    networks = [net.id for net in nets],
                    target_names = list(recipes.keys()),
                    feature_names = feature_names
                )
                print('Created MultiNetwork with id: {:d}'.format(multinet.id))
            else:
                print('Network with Networks {!s} already exists with id: {:d}. Skipping MultiNetwork creation'.format([net.id for net in nets], multinet.id))
        return multinet

    def to_QuaLiKizNDNN(self, cached_purenets=None, **nn_kwargs):
        if cached_purenets is None:
            cached_purenets= {}
        if not cached_purenets:
            pure_params = self.pure_network_params.get()
            json_dict = pure_params.network_json.get().network_json
        else:
            json_dict = cached_purenets[self.id]['network_json']
        qlknet = QuaLiKizNDNN(json_dict, **nn_kwargs)
        return qlknet
    #cached_purenets[self.id]['network_json'].to_QuaLiKizNDNN(**nn_kwargs)

    def to_QuaLiKizComboNN(self, combo_kwargs=None, cached_purenets=None, **nn_kwargs):
        if combo_kwargs is None:
            combo_kwargs = {}
        if cached_purenets is None:
            cached_purenets= {}

        network_ids = self.networks
        # Hackish way to get all pure children of this network
        if not all(id in cached_purenets for id in self.networks):
            subq = self.get_recursive_subquery('cost_l2_scale')
            subq = subq.having(SQL('root') == self.id).dicts()
            if len(subq) == 1:
                res = subq.get()
                pure_children = res['pure_children']
                nets = (Network.select()
                        .where(Network.id.in_(pure_children))
                        )
                pures = PureNetworkParams.select()
                jsons = NetworkJSON.select()
                pures_and_nets = prefetch(nets, pures, jsons)
                cached_purenets.update({network.id: {'net': network,
                                                     'pure_net':network.pure_network_params[0],
                                                     'network_json': network.pure_network_params[0].network_json[0].network_json}
                                        for network in pures_and_nets})

        networks = []
        if all(id in cached_purenets for id in self.networks):
            for id in self.networks:
                qlknet = cached_purenets[id]['net'].to_QuaLiKizNN(cached_purenets=cached_purenets)
                networks.append(qlknet)
        else:
            nets = {net.id: net for net in Network.select().where(Network.id.in_(network_ids))}
            for id in self.networks:
                qlknet = nets[id].to_QuaLiKizNN(cached_purenets=cached_purenets)
                networks.append(qlknet)

        recipe = self.recipe
        for ii in range(len(network_ids)):
            recipe = re.sub('nn(\d*)', 'args[\\1]', recipe)
        exec('def combo_func(*args): return ' + recipe, globals())
        return QuaLiKizComboNN(self.target_names, networks, combo_func, **combo_kwargs)

    def to_QuaLiKizNN(self, cached_purenets=None, combo_kwargs=None, **nn_kwargs):
        if combo_kwargs is None:
            combo_kwargs = {}
        if self.networks is None:
            net = self.to_QuaLiKizNDNN(cached_purenets=cached_purenets, **nn_kwargs)
        else:
            net = self.to_QuaLiKizComboNN(cached_purenets=cached_purenets, combo_kwargs=combo_kwargs, **nn_kwargs)
        return net

    @classmethod
    def calc_op(cls, column):
        query = (cls.select(
                            ComboNetwork,
                            ComboNetwork.id.alias('combo_id'),
                            fn.ARRAY_AGG(getattr(Hyperparameters, column), coerce=False).alias(column))
                 .join(Network, on=(Network.id == fn.ANY(ComboNetwork.networks)))
                 .join(Hyperparameters, on=(Network.id == Hyperparameters.network_id))
                 .group_by(cls.id)
        )
        return query

    def get_pure_children(self):
        if self.networks is None:
            return [self]
        else:
            subq = self.get_recursive_subquery('cost_l2_scale')
            subq = subq.having(SQL('root') == self.id)
            if len(subq) == 1:
                pure_ids = subq.get().pure_children
                query = Network.select().where(Network.id.in_(pure_ids))
                return [net for net in query]
            else:
                raise

class PureNetworkParams(BaseModel):
    network = ForeignKeyField(Network, related_name='pure_network_params', unique=True)
    filter = ForeignKeyField(Filter, related_name='pure_network_params')
    train_script = ForeignKeyField(TrainScript, related_name='pure_network_params')
    feature_prescale_bias = HStoreField()
    feature_prescale_factor = HStoreField()
    target_prescale_bias = HStoreField()
    target_prescale_factor = HStoreField()
    feature_min = HStoreField()
    feature_max = HStoreField()
    target_min = HStoreField()
    target_max = HStoreField()
    timestamp = DateTimeField(constraints=[SQL('DEFAULT now()')])

    def download_raw(self):
        root_dir = 'Network_' + str(self.network_id)
        os.mkdir(root_dir)
        network_json = self.network_json.get()
        with open(os.path.join(root_dir, 'settings.json'), 'w') as settings_file:
            json.dump(network_json.settings_json, settings_file,
                       sort_keys=True, indent=4)
        with open(os.path.join(root_dir, 'nn.json'), 'w') as network_file:
            json.dump(network_json.network_json, network_file,
                       sort_keys=True, indent=4)
        with open(os.path.join(root_dir, 'train_NDNN.py'), 'w') as train_file:
            train_file.writelines(self.train_script.get().script)

    @classmethod
    def find_similar_topology_by_settings(cls, settings_path):
        with open(settings_path) as file_:
            json_dict = json.load(file_)
            cls.find_similar_topology_by_values(
                                                json_dict['hidden_neurons'],
                                                json_dict['hidden_activation'],
                                                json_dict['output_activation'],
                                                train_dim=json_dict['train_dim'])
        return query

    @classmethod
    def find_similar_topology_by_id(cls, pure_network_params_id, match_train_dim=True):
        query = (cls
                 .select(
                         Hyperparameters.hidden_neurons,
                         Hyperparameters.hidden_activation,
                         Hyperparameters.output_activation)
                 .where(cls.id == pure_network_params_id)
                 .join(Hyperparameters)
        )

        train_dim, = (cls
                 .select(
                         Network.target_names)
                 .where(cls.id == pure_network_params_id)
                 .join(Network)
                 ).tuples().get()
        if match_train_dim is not True:
            train_dim = None
        query = cls.find_similar_topology_by_values(*query.tuples().get(), train_dim=train_dim)
        query = query.where(cls.id != pure_network_params_id)
        return query

    @classmethod
    def find_similar_topology_by_values(cls, hidden_neurons, hidden_activation, output_activation, train_dim=None):
        query = (cls.select()
                 .join(Hyperparameters)
                 .where(Hyperparameters.hidden_neurons ==
                        hidden_neurons)
                 .where(Hyperparameters.hidden_activation ==
                        hidden_activation)
                 .where(Hyperparameters.output_activation ==
                        output_activation))

        if train_dim is not None:
            query = (query.where(Network.target_names == train_dim)
                     .switch(cls)
                     .join(Network))
        return query

    @classmethod
    def find_similar_networkpar_by_settings(cls, settings_path):
        with open(settings_path) as file_:
            json_dict = json.load(file_)

        query = cls.find_similar_networkpar_by_values(json_dict['train_dim'],
                                                    json_dict['goodness'],
                                                    json_dict['cost_l2_scale'],
                                                    json_dict['cost_l1_scale'],
                                                    json_dict['early_stop_after'],
                                                    json_dict['early_stop_measure'])
        return query

    @classmethod
    def find_similar_networkpar_by_id(cls, pure_network_params_id, ignore_pars=None, match_train_dim=True):
        if ignore_pars is None:
            ignore_pars = []
        networkpars = ['goodness', 'cost_l2_scale', 'cost_l1_scale', 'early_stop_measure', 'early_stop_after']
        select_pars = [getattr(Hyperparameters, par) for par in networkpars if par not in ignore_pars]

        query = (cls
                 .select(*select_pars)
                 .where(cls.id == pure_network_params_id)
                 .join(Hyperparameters)
        )

        filter_id, train_dim = (cls
                 .select(cls.filter_id,
                         Network.target_names)
                 .where(cls.id == pure_network_params_id)
                 .join(Network)
                 ).tuples().get()
        if match_train_dim is not True:
            train_dim = None

        query = cls.find_similar_networkpar_by_values(query.dicts().get(), filter_id=filter_id, train_dim=train_dim)
        query = query.where(cls.id != pure_network_params_id)
        return query

    @classmethod
    def find_similar_networkpar_by_values(cls, networkpar_dict, filter_id=None, train_dim=None):
        # TODO: Add new hyperparameters here?
        query = (cls.select().join(Hyperparameters))
        for parname, val in networkpar_dict.items():
            attr = getattr(Hyperparameters, parname)
            if isinstance(val, float):
                attr = attr.cast('numeric')
            query = query.where(attr == val)

        if train_dim is not None:
            query = (query.where(Network.target_names ==
                                 train_dim)
                     .switch(cls)
                     .join(Network)
            )

        if filter_id is not None:
            query = query.where(cls.filter_id ==
                                filter_id)
        else:
            print('Warning! Not filtering on filter_id')
        return query

    #@classmethod
    #def find_similar_networkpar_by_settings(cls, settings_path):
    #    with open(settings_path) as file_:
    #        json_dict = json.load(file_)

    #    query = cls.find_similar_networkpar_by_values(json_dict['train_dim'],
    #                                                json_dict['goodness'],
    #                                                json_dict['cost_l2_scale'],
    #                                                json_dict['cost_l1_scale'],
    #                                                json_dict['early_stop_measure'])
    #    return query

    @classmethod
    def find_similar_trainingpar_by_id(cls, pure_network_params_id):
        query = (cls
                 .select(Network.target_names,
                         Hyperparameters.minibatches,
                         Hyperparameters.optimizer,
                         Hyperparameters.standardization,
                         Hyperparameters.early_stop_after)
                 .where(cls.id == pure_network_params_id)
                 .join(Hyperparameters)
                 .join(Network)
        )

        filter_id = (cls
                 .select(cls.filter_id)
                 .where(cls.id == cls.network_id)
                 ).tuples().get()[0]
        query = cls.find_similar_trainingpar_by_values(*query.tuples().get())
        query = query.where(cls.id != pure_network_params_id)
        return query

    @classmethod
    def find_similar_trainingpar_by_values(cls, train_dim, minibatches, optimizer, standardization, early_stop_after):
        query = (cls.select()
                 .where(Network.target_names == AsIs(train_dim))
                 .join(Hyperparameters)
                 .where(Hyperparameters.minibatches == minibatches)
                 .where(Hyperparameters.optimizer == optimizer)
                 .where(Hyperparameters.standardization == standardization)
                 .where(Hyperparameters.early_stop_after == early_stop_after)
                 )
        return query


    @classmethod
    def from_folders(cls, pwd, **kwargs):
        for path_ in os.listdir(pwd):
            path_ = os.path.join(pwd, path_)
            if os.path.isdir(path_):
                try:
                    Network.from_folder(path_, **kwargs)
                except IOError:
                    print('Could not parse', path_, 'is training done?')

    @classmethod
    @db.atomic()
    def from_folder(cls, pwd):
        script_file = os.path.join(pwd, 'train_NDNN.py')
        #with open(script_file, 'r') as script:
        #    script = script.read()
        train_script = TrainScript.from_file(script_file)

        json_path = os.path.join(pwd, 'nn.json')
        nn = QuaLiKizNDNN.from_json(json_path)
        with open(json_path) as file_:
            json_dict = json.load(file_)
            dict_ = {}
            for name in ['feature_prescale_bias', 'feature_prescale_factor',
                         'target_prescale_bias', 'target_prescale_factor',
                         'feature_names', 'feature_min', 'feature_max',
                         'target_names', 'target_min', 'target_max']:
                attr = getattr(nn, '_' + name)
                if 'names' in name:
                    dict_[name] = list(attr)
                else:
                    dict_[name] = {str(key): str(val) for key, val in attr.items()}

        dict_['train_script'] = train_script
        net_dict = {'feature_names': dict_.pop('feature_names'),
                    'target_names': dict_.pop('target_names')}

        with open(os.path.join(pwd, 'settings.json')) as file_:
            settings = json.load(file_)

        dict_['filter_id'] = Filter.find_by_path_name(settings['dataset_path'])
        network = Network.create(**net_dict)
        dict_['network'] = network
        pure_network_params = PureNetworkParams.create(**dict_)
        pure_network_params.save()
        hyperpar = Hyperparameters.from_settings(pure_network_params, settings)
        hyperpar.save()
        if settings['optimizer'] == 'lbfgs':
            optimizer = LbfgsOptimizer(
                pure_network_params=pure_network_params,
                maxfun=settings['lbfgs_maxfun'],
                maxiter=settings['lbfgs_maxiter'],
                maxls=settings['lbfgs_maxls'])
        elif settings['optimizer'] == 'adam':
            optimizer = AdamOptimizer(
                pure_network_params=pure_network_params,
                learning_rate=settings['learning_rate'],
                beta1=settings['adam_beta1'],
                beta2=settings['adam_beta2'])
        elif settings['optimizer'] == 'adadelta':
            optimizer = AdadeltaOptimizer(
                pure_network_params=pure_network_params,
                learning_rate=settings['learning_rate'],
                rho=settings['adadelta_rho'])
        elif settings['optimizer'] == 'rmsprop':
            optimizer = RmspropOptimizer(
                pure_network_params=pure_network_params,
                learning_rate=settings['learning_rate'],
                decay=settings['rmsprop_decay'],
                momentum=settings['rmsprop_momentum'])
        optimizer.save()

        activations = settings['hidden_activation'] + [settings['output_activation']]
        for ii, layer in enumerate(nn.layers):
            nwlayer = NetworkLayer.create(
                pure_network_params=pure_network_params,
                weights = np.float32(layer._weights).tolist(),
                biases = np.float32(layer._biases).tolist(),
                activation = activations[ii])

        NetworkMetadata.from_dict(json_dict['_metadata'], pure_network_params)
        TrainMetadata.from_folder(pwd, pure_network_params)

        network_json = NetworkJSON.create(
            pure_network_params=pure_network_params,
            network_json=json_dict,
            settings_json=settings)
        return network

    #def to_QuaLiKizNDNN(self, **nn_kwargs):
    #    json_dict = self.network_json.get().network_json
    #    nn = QuaLiKizNDNN(json_dict, **nn_kwargs)
    #    return nn

    #to_QuaLiKizNN = to_QuaLiKizNDNN

    def to_matlab_dict(self):
        js = self.network_json.get().network_json
        matdict = nn_dict_to_matlab(js)
        return matdict

    def to_matlab(self):
        io.savemat(str(self.id) + '.mat', self.to_matlab_dict())

    def summarize(self):
        net = self.select().get()
        print({'target_names':     net.target_names,
               'rms_test':         net.network_metadata.get().rms_test,
               'rms_train':        net.network_metadata.get().rms_train,
               'rms_validation':   net.network_metadata.get().rms_validation,
               'epoch':            net.network_metadata.get().epoch,
               'train_time':       net.train_metadata.get().walltime[-1],
               'hidden_neurons':   net.hyperparameters.get().hidden_neurons,
               'standardization':  net.hyperparameters.get().standardization,
               'cost_l2_scale':    net.hyperparameters.get().cost_l2_scale,
               'early_stop_after': net.hyperparameters.get().early_stop_after}
        )


class NetworkJSON(BaseModel):
    pure_network_params = ForeignKeyField(PureNetworkParams, related_name='network_json', unique=True)
    network_json = BinaryJSONField()
    settings_json = BinaryJSONField()

class NetworkLayer(BaseModel):
    pure_network_params = ForeignKeyField(PureNetworkParams, related_name='network_layer')
    weights = ArrayField(FloatField)
    biases = ArrayField(FloatField)
    activation = TextField()

class NetworkMetadata(BaseModel):
    pure_network_params = ForeignKeyField(PureNetworkParams, related_name='network_metadata', unique=True)
    epoch = IntegerField()
    best_epoch = IntegerField()
    rms_test = FloatField(null=True)
    rms_train = FloatField(null=True)
    rms_validation = FloatField()
    rms_validation_descaled = FloatField(null=True)
    loss_test = FloatField(null=True)
    loss_train = FloatField(null=True)
    loss_validation = FloatField()
    metadata = HStoreField()

    @classmethod
    @db.atomic()
    def from_dict(cls, json_dict, pure_network_params):
        stringified = {str(key): str(val) for key, val in json_dict.items()}
        try:
            rms_train = json_dict['rms_train']
            loss_train = json_dict['loss_train']
        except KeyError:
            loss_train = rms_train = None
        try:
            loss_test = json_dict['loss_test']
            rms_test = json_dict['loss_test']
        except KeyError:
            rms_test = loss_test = None
        try:
            rms_validation_descaled = json_dict['rms_validation_descaled']
        except KeyError:
            rms_validation_descaled = None
        network_metadata = NetworkMetadata(
            pure_network_params=pure_network_params,
            epoch=json_dict['epoch'],
            best_epoch=json_dict['best_epoch'],
            rms_train=rms_train,
            rms_validation=json_dict['rms_validation'],
            rms_validation_descaled=rms_validation_descaled,
            rms_test=rms_test,
            loss_train=loss_train,
            loss_validation=json_dict['loss_validation'],
            loss_test=loss_test,
            metadata=stringified
        )
        network_metadata.save()
        return network_metadata

class TrainMetadata(BaseModel):
    pure_network_params = ForeignKeyField(PureNetworkParams, related_name='train_metadata')
    set = TextField(choices=['train', 'test', 'validation'])
    step =         ArrayField(IntegerField)
    epoch =        ArrayField(IntegerField)
    walltime =     ArrayField(FloatField)
    loss =         ArrayField(FloatField)
    mse =          ArrayField(FloatField)
    mabse =        ArrayField(FloatField, null=True)
    l1_norm =      ArrayField(FloatField, null=True)
    l2_norm =      ArrayField(FloatField, null=True)
    hostname = TextField()

    @classmethod
    @db.atomic()
    def from_folder(cls, pwd, pure_network_params):
        train_metadatas = None
        for name in cls.set.choices:
            train_metadatas = []
            try:
                with open(os.path.join(pwd, name + '_log.csv')) as file_:
                    df = pd.DataFrame.from_csv(file_)
            except IOError:
                pass
            else:
                # TODO: Only works on debian-like
                train_metadata = TrainMetadata(
                    pure_network_params=pure_network_params,
                    set=name,
                    step=[int(x) for x in df.index],
                    epoch=[int(x) for x in df['epoch']],
                    walltime=df['walltime'],
                    loss=df['loss'],
                    mse=df['mse'],
                    mabse=df['mabse'],
                    l1_norm=df['l1_norm'],
                    l2_norm=df['l2_norm'],
                    hostname=socket.gethostname()
                )
                train_metadata.save()
                train_metadatas.append(train_metadata)
        return train_metadatas


class Hyperparameters(BaseModel):
    pure_network_params = ForeignKeyField(PureNetworkParams, related_name='hyperparameters', unique=True)
    hidden_neurons = ArrayField(IntegerField)
    hidden_activation = ArrayField(TextField)
    output_activation = TextField()
    standardization = TextField()
    goodness = TextField()
    drop_chance = FloatField()
    optimizer = TextField()
    cost_l2_scale = FloatField()
    cost_l1_scale = FloatField()
    early_stop_after = FloatField()
    early_stop_measure = TextField()
    minibatches = IntegerField()
    drop_outlier_above = FloatField()
    drop_outlier_below = FloatField()
    validation_fraction = FloatField()
    dtype = TextField()

    @classmethod
    def from_settings(cls, pure_network_params, settings):
        hyperpar = cls(pure_network_params=pure_network_params,
                       hidden_neurons=settings['hidden_neurons'],
                       hidden_activation=settings['hidden_activation'],
                       output_activation=settings['output_activation'],
                       standardization=settings['standardization'],
                       goodness=settings['goodness'],
                       drop_chance=settings['drop_chance'],
                       optimizer=settings['optimizer'],
                       cost_l2_scale=settings['cost_l2_scale'],
                       cost_l1_scale=settings['cost_l1_scale'],
                       early_stop_after=settings['early_stop_after'],
                       early_stop_measure=settings['early_stop_measure'],
                       minibatches=settings['minibatches'],
                       drop_outlier_above=settings['drop_outlier_above'],
                       drop_outlier_below=settings['drop_outlier_below'],
                       validation_fraction=settings['validation_fraction'],
                       dtype=settings['dtype'])
        return hyperpar


class LbfgsOptimizer(BaseModel):
    pure_network_params = ForeignKeyField(PureNetworkParams, related_name='lbfgs_optimizer', unique=True)
    maxfun = IntegerField()
    maxiter = IntegerField()
    maxls = IntegerField()

class AdamOptimizer(BaseModel):
    pure_network_params = ForeignKeyField(PureNetworkParams, related_name='adam_optimizer', unique=True)
    learning_rate = FloatField()
    beta1 = FloatField()
    beta2 = FloatField()

class AdadeltaOptimizer(BaseModel):
    pure_network_params = ForeignKeyField(PureNetworkParams, related_name='adadelta_optimizer', unique=True)
    learning_rate = FloatField()
    rho = FloatField()

class RmspropOptimizer(BaseModel):
    pure_network_params = ForeignKeyField(PureNetworkParams, related_name='rmsprop_optimizer', unique=True)
    learning_rate = FloatField()
    decay = FloatField()
    momentum = FloatField()

class Postprocess(BaseModel):
    network         = ForeignKeyField(Network, related_name='postprocess')
    filter          = ForeignKeyField(Filter, related_name='postprocess')
    rms             = ArrayField(FloatField)
    leq_bound       = FloatField()
    less_bound      = FloatField()

class PostprocessSlice(BaseModel):
    network = ForeignKeyField(Network, related_name='postprocess_slice', null=True)
    thresh_abs_mis_median       = ArrayField(FloatField)
    thresh_abs_mis_95width      = ArrayField(FloatField)
    thresh_rel_mis_median       = ArrayField(FloatField)
    thresh_rel_mis_95width      = ArrayField(FloatField)
    no_thresh_frac              = ArrayField(FloatField)
    pop_abs_mis_median          = ArrayField(FloatField)
    pop_abs_mis_95width         = ArrayField(FloatField)
    pop_rel_mis_median          = ArrayField(FloatField)
    pop_rel_mis_95width         = ArrayField(FloatField)
    no_pop_frac                 = ArrayField(FloatField)
    wobble_tot                  = ArrayField(FloatField)
    wobble_unstab               = ArrayField(FloatField)
    wobble_qlkunstab            = ArrayField(FloatField)
    frac                        = FloatField()
    store_name                  = TextField()
    dual_thresh_mismatch_median = FloatField(null=True)
    dual_thresh_mismatch_95width= FloatField(null=True)
    no_dual_thresh_frac         = FloatField(null=True)

class PostprocessSlice_9D(PostprocessSlice):
    pass

### A few convinience functions to select a network by (nested) cost_l2_scale
def select_from_candidate_query(candidates_query):
    if len(candidates_query) < 1:
        raise Exception('No candidates found')
    elif len(candidates_query) == 1:
        return candidates_query.get()
    else:
        return Network.pick_candidate(candidates_query).get()

def get_from_cost_l2_scale_array(target_name, cost_l2_scale_array):
    subq = Network.get_recursive_subquery('cost_l2_scale')
    subq2 = (subq
             .having(fn.ARRAY_AGG(SQL("DISTINCT childq.cost_l2_scale"), coerce=False) == SQL("'" + cost_l2_scale_array + "'"))
             .having(SQL("net.target_names = '{" + target_name + "}'"))
             )
    candidate_ids = [el['root'] for el in subq2.dicts()]
    candidates_query = Network.select().where(Network.id.in_(candidate_ids))
    return select_from_candidate_query(candidates_query)

def get_pure_from_cost_l2_scale(target_name, cost_l2_scale):
    candidates_query = (Network.select()
                        .where(Hyperparameters.cost_l2_scale.cast('numeric') == cost_l2_scale)
                        .where(Network.target_names == [target_name])
                        .join(PureNetworkParams)
                        .join(Hyperparameters)
                        )
    return select_from_candidate_query(candidates_query)
def create_schema():
    db.execute_sql('SET ROLE developer;')
    db.execute_sql('CREATE SCHEMA develop AUTHORIZATION developer;')
    db.execute_sql('ALTER DEFAULT PRIVILEGES IN SCHEMA develop GRANT ALL ON TABLES TO developer;')

def create_tables():
    db.execute_sql('SET ROLE developer;')
    db.create_tables([Filter, Network, PureNetworkParams, NetworkJSON, NetworkLayer, NetworkMetadata, TrainMetadata, Hyperparameters, LbfgsOptimizer, AdamOptimizer, AdadeltaOptimizer, RmspropOptimizer, TrainScript, PostprocessSlice, Postprocess])

def purge_tables():
    clsmembers = inspect.getmembers(sys.modules[__name__], lambda member: inspect.isclass(member) and member.__module__ == __name__)
    for name, cls in clsmembers:
        if name != BaseModel:
            try:
                db.drop_table(cls, cascade=True)
            except ProgrammingError:
                db.rollback()

#def any_element_in_list(cls, column, tags):
#    subquery = (cls.select(cls.id.alias('id'),
#                               fn.unnest(getattr(cls, column)).alias('unnested_tags'))
#                .alias('subquery'))
#    tags_filters = [subquery.c.unnested_tags.contains(tag) for tag in tags]
#    tags_filter = reduce(operator.or_, tags_filters)
#    query = (cls.select()
#             .join(subquery, on=subquery.c.id == cls.id)
#             .where(tags_filter)
#             # gets rid of duplicates
#             .group_by(cls.id)
#    )
#    return query
#
#def no_elements_in_list(cls, column, tags, fields=None):
#    subquery = (cls.select(cls.id.alias('id'),
#                               fn.unnest(getattr(cls, column)).alias('unnested_tags'))
#                .alias('subquery'))
#    tags_filters = [subquery.c.unnested_tags.contains(tag) for tag in tags]
#    tags_filter = reduce(operator.or_, tags_filters)
#    if not fields:
#        fields = Network._meta.sorted_fields
#    query = (cls.select(fields)
#             .join(subquery, on=subquery.c.id == cls.id)
#             .where(~tags_filter)
#             # gets rid of duplicates
#             .group_by(cls.id)
#    )
#    return query

def create_views():
    """
    CREATE VIEW
    SUMMARY AS
    SELECT A.id, target_names, hidden_neurons, standardization, cost_l2_scale, early_stop_after, best_rms_test, best_rms_validation, best_rms_train, final_rms_validation, final_rms_train, walltime, hostname FROM
    (
    SELECT network.id, network.target_names, hyperparameters.hidden_neurons, hyperparameters.standardization, hyperparameters.cost_l2_scale, hyperparameters.early_stop_after, networkmetadata.rms_test as best_rms_test, networkmetadata.rms_validation as best_rms_validation, networkmetadata.rms_train as best_rms_train
    FROM network
    INNER JOIN hyperparameters
    ON network.id = hyperparameters.network_id
    INNER JOIN networkmetadata
    ON network.id = networkmetadata.network_id
    ) A
    INNER JOIN
    (
    SELECT network.id AS id_B, sqrt(trainmetadata.mse[array_length(trainmetadata.mse, 1)]) as final_rms_validation
    FROM network
    INNER JOIN trainmetadata
    ON network.id = trainmetadata.network_id
    WHERE trainmetadata.set = 'validation'
    ) B
    ON A.id = B.id_B
    INNER JOIN
    (
    SELECT network.id AS id_C, sqrt(trainmetadata.mse[array_length(trainmetadata.mse, 1)]) as final_rms_train, trainmetadata.walltime[array_length(trainmetadata.walltime, 1)], trainmetadata.hostname
    FROM network
    INNER JOIN trainmetadata
    ON network.id = trainmetadata.network_id
    WHERE trainmetadata.set = 'train'
    ) C
    ON A.id = C.id_C
    """
    """
     DROP VIEW SUMMARY_LOSS;
CREATE VIEW
    SUMMARY_LOSS AS
    SELECT A.id, target_names, hidden_neurons, standardization, cost_l2_scale, early_stop_after, best_rms_test,  best_rms_validation, l2_norm_validation, walltime, hostname FROM
    (
    SELECT network.id, network.target_names, hyperparameters.hidden_neurons, hyperparameters.standardization, hyperparameters.cost_l2_scale, hyperparameters.early_stop_after, networkmetadata.rms_test as best_rms_test, networkmetadata.rms_validation as best_rms_validation
    FROM network
    INNER JOIN hyperparameters
    ON network.id = hyperparameters.network_id
    INNER JOIN networkmetadata
    ON network.id = networkmetadata.network_id
    WHERE hyperparameters.early_stop_measure = 'loss'
    ) A
    INNER JOIN
    (
    SELECT network.id AS id_C, trainmetadata.l2_norm[networkmetadata.best_epoch + 1] as l2_norm_validation, trainmetadata.walltime[array_length(trainmetadata.walltime, 1)], trainmetadata.hostname
    FROM network
    INNER JOIN trainmetadata
    ON network.id = trainmetadata.network_id
    INNER JOIN networkmetadata
    ON network.id = networkmetadata.network_id
    WHERE trainmetadata.set = 'validation'
    ) C
    ON A.id = C.id_C
"""

"""
Avg l2 multinetwork:
SELECT multinetwork.id as multi_id, multinetwork.target_names, AVG(cost_l2_scale) AS cost_l2_scale
FROM "multinetwork"
JOIN combonetwork ON (combo_network_id = combonetwork.id) OR (combonetwork.id = ANY (combo_network_partners))
JOIN network ON (network.id = ANY (combonetwork.networks))
JOIN hyperparameters ON (network.id = hyperparameters.network_id)
GROUP BY multinetwork.id
ORDER BY multi_id
"""

if __name__ == '__main__':
    from IPython import embed
    embed()
#purge_tables()
#create_tables()
#create_views()
#Network.from_folder('finished_nns_filter2/efiITG_GB_filter2', filter_id=3)
