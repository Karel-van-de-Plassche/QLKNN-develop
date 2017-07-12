from peewee import *
from peewee import FloatField, FloatField, ProgrammingError, IntegerField
import numpy as np
import inspect
import sys
from playhouse.postgres_ext import PostgresqlExtDatabase, ArrayField, BinaryJSONField, JSONField, HStoreField
from IPython import embed
import scipy as sc
from scipy import io
import os
from run_model import QuaLiKizNDNN
import json

db = PostgresqlExtDatabase(database='nndb', host='gkdb.org')
class BaseModel(Model):
    """A base model that will use our Postgresql database"""
    class Meta:
        database = db
        schema = 'develop'

class Filter(BaseModel):
    script = TextField()
    description = TextField()

class Network(BaseModel):
    filter = ForeignKeyField(Filter, related_name='filter', null=True)
    prescale_bias = HStoreField()
    prescale_factor = HStoreField()
    feature_names = ArrayField(TextField)
    feature_min = HStoreField()
    feature_max = HStoreField()
    target_names = ArrayField(TextField)
    target_min = HStoreField()
    target_max = HStoreField()
    #weights = ArrayField(FloatField, dimensions=3)
    #biases = ArrayField(FloatField, dimensions=2)
    json = JSONField()

    @classmethod
    def from_folder(cls, pwd):
        json_path = os.path.join(pwd, 'nn.json')
        nn = QuaLiKizNDNN.from_json(json_path)
        with open(json_path) as file_:
            json_dict = json.load(file_)
        dict_ = {'json': json_dict}
        for name in ['prescale_bias', 'prescale_factor',
                     'feature_names', 'feature_min', 'feature_max',
                     'target_names', 'target_min', 'target_max']:
            attr = getattr(nn, name)
            if 'names' in name:
                dict_[name] = list(attr)
            else:
                dict_[name] = {str(key): str(val) for key, val in attr.items()}

        network = Network(**dict_)
        network.save()
        for layer in nn.layers:
            nwlayer = NetworkLayer(network = network,
                                   weights = layer.weight.tolist(),
                                   biases = layer.bias.tolist(),
                                   activation = nn._metadata['activation'])
            nwlayer.save()

        with open(os.path.join(pwd, 'settings.json')) as file_:
            settings = json.load(file_)
        hyperpar = Hyperparameters(network=network,
                                   hidden_neurons=settings['hidden_neurons'],
                                   scaling=settings['scaling'],
                                   cost_l2_scale=settings['cost_l2_scale'],
                                   early_stop_after=settings['early_stop_after'])
        hyperpar.save()
        if settings['optimizer'] == 'lbfgs':
            optimizer = LbfgsOptimizer(hyperparameters=hyperpar,
                                       maxfun=settings['lbfgs_maxfun'],
                                       maxiter=settings['lbfgs_maxiter'],
                                       maxls=settings['lbfgs_maxls'])
        optimizer.save()
        embed()

class NetworkLayer(BaseModel):
    network = ForeignKeyField(Network)
    weights = ArrayField(FloatField)
    biases = ArrayField(FloatField)
    activation = TextField()

class NetworkMetadata(BaseModel):
    network = ForeignKeyField(Network)
    metadata =  HStoreField()

class TrainMetadata(BaseModel):
    network = ForeignKeyField(Network)
    set = TextField(choices=['train', 'test', 'validation'])
    step = IntegerField()
    epoch = IntegerField()
    walltime = FloatField()
    loss = FloatField()
    mse = FloatField()

class Hyperparameters(BaseModel):
    network = ForeignKeyField(Network)
    hidden_neurons = ArrayField(IntegerField)
    scaling = TextField()
    cost_l2_scale = FloatField()
    early_stop_after = FloatField()

class LbfgsOptimizer(BaseModel):
    hyperparameters = ForeignKeyField(Hyperparameters)
    maxfun = IntegerField()
    maxiter = IntegerField()
    maxls = IntegerField()

def create_tables():
    db.execute_sql('SET ROLE developer')
    db.create_tables([Filter, Network, NetworkLayer, NetworkMetadata, TrainMetadata, Hyperparameters, LbfgsOptimizer])

def purge_tables():
    clsmembers = inspect.getmembers(sys.modules[__name__], lambda member: inspect.isclass(member) and member.__module__ == __name__)
    for name, cls in clsmembers:
        if name != BaseModel:
            try:
                db.drop_table(cls, cascade=True)
            except ProgrammingError:
                db.rollback()


purge_tables()
create_tables()
Network.from_folder('nns/efeITG_GB_3')
