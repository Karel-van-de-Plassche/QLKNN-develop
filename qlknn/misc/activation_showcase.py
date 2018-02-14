import tensorflow as tf
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import pandas as pd
from IPython import embed

def qualikiz_sigmoid(x, name=""):
    dtype = x.dtype.name
    return tf.divide(tf.constant(2, dtype),
                     tf.add(tf.constant(1, dtype),
                            tf.exp(tf.multiply(tf.constant(-2, dtype), x)))) - tf.constant(1, dtype)
sess = tf.InteractiveSession()
x = np.linspace(-2, 2, 50)
plt.figure()
plt.plot(x, tf.nn.relu(x).eval(), label='relu')
#plt.plot(x, tf.nn.relu6(x).eval(), label='relu6')
plt.plot(x, tf.nn.elu(x).eval(), '.', label='elu')
plt.plot(x, tf.nn.softplus(x).eval(), label='softplus')
plt.plot(x, tf.nn.softsign(x).eval(), label='softsign')
plt.plot(x, tf.nn.sigmoid(x).eval(), label='sigmoid')
plt.plot(x, tf.nn.tanh(x).eval(), label='tanh')
plt.plot(x, qualikiz_sigmoid(x).eval(), label='QuaLiKiz')
plt.legend()
plt.show()
