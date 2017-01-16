import argparse
import os
import sys

import tensorflow as tf
import xarray as xr
from IPython import embed
from collections import OrderedDict
from itertools import product

def _int64_feature(value):
      return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float32_feature(value):
      return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to(ds, filename):
    scan_dims = OrderedDict([(name, ds[name].data) for name in ds.dims if name not in ['nions', 'numsols', 'kthetarhos']])
    writer = tf.python_io.TFRecordWriter(filename)

    for jj, foo in enumerate(product(*scan_dims.values())):
        print(jj)
        bar = dict(zip(scan_dims.keys(), foo))
        barf = ds.sel(**bar)
        example_dict = {}
        for name, barfoo in barf.data_vars.items():
            if barfoo.dims == ():
                example_dict[name] = _float32_feature(float(barfoo))
            elif barfoo.dims == ('nions', ):
                for ii, val in enumerate(barfoo.data):
                    example_dict[name+str(ii)] = _float32_feature(float(val))
        example = tf.train.Example(features=tf.train.Features(feature=example_dict))
        writer.write(example.SerializeToString())
    writer.close()

    
    #ds = ds.stack(dimx=scan_dims)
    #ds = ds.transpose('dimx', 'nions', 'kthetarhos', 'numsols')
    #ele_like = OrderedDict([(var_name, var_value) for var_name, var_value in ds.items() if var_value.dims==('dimx', ) and var_value.name !='dimx'])
    #ion_like = OrderedDict([(var_name, var_value) for var_name, var_value in ds.items() if var_value.dims==('dimx', 'nions') and var_value.name !='nions'])
    #for foo in zip(*ele_like.values()):
    #    pass
    #    print([float(fool) for fool in foo])
    #    break
    embed()

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    embed()
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })
    
    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape([mnist.IMAGE_PIXELS])
    
    # OPTIONAL: Could reshape into a 28x28 image and apply distortions
    # here.  Since we are not applying any distortions in this
    # example, and the next step expects the image to be flattened
    # into a vector, we don't bother.
    
    # Convert from [0, 255] -> [-0.5, 0.5] floats.
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    
    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(features['label'], tf.int32)
    
    return image, label

ds = xr.open_dataset('/mnt/hdd/4D.nc')
#ds = xr.open_dataset('/mnt/hdd/Zeff_combined.nc')
#convert_to(ds, 'hi')
read_and_decode('hi')
