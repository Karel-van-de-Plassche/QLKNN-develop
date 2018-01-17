from ctypes import *
from ctypes.util import find_library
from IPython import embed
import numpy as np
mkl = np.ctypeslib.load_library('libmkl_rt', '/opt/intel/compilers_and_libraries/linux/mkl/lib/intel64/')
CblasRowMajor  = c_int(101)
CblasColMajor  = c_int(102)
CblasNoTrans   = c_int(111)
CblasTrans     = c_int(112)
CblasConjTrans = c_int(113)
#print(mkl.vdFmax(5))
alpha = beta = c_double(1)

class Layer():
    """ A single (hidden) NN layer
    A hidden NN layer is just does

        output = activation(weight * input + bias)

        Where weight is generally a matrix; output, input and bias a vector
        and activation a (sigmoid) function.
        """

    def __init__(self, weight, bias, activation):
        self._weights = weight
        self._biases = np.atleast_2d(bias)
        self._activation = activation

    def apply(self, input, output=None):
        A = input
        B = self._weights
        m, k = A.shape
        _, n = self._weights.shape
        C = np.tile(self._biases, [m, 1])
        #np.zeros_like(self.biases)
        mkl.cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, k, B, n ,beta, C, n)
        if self._activation == 'tanh':
            mkl.vdTanh(n*m, C, C)
        return C

    def shape(self):
        return self.weight.shape

    def __str__(self):
        return ('NNLayer shape ' + str(self.shape()))

mkl.vdTanh.argtypes = [
    c_int,
    np.ctypeslib.ndpointer(float, ndim=2,
                           flags='aligned, contiguous'),
    np.ctypeslib.ndpointer(float, ndim=2,
                           flags='aligned, contiguous,'\
                           'writeable'),
]
mkl.vdTanh.restype = None
mkl.cblas_dgemm.restype = None
mkl.cblas_dgemm.argtypes = [
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    c_double,
    np.ctypeslib.ndpointer(float, ndim=2,
                           flags='aligned, contiguous'),
    c_int,
    np.ctypeslib.ndpointer(float, ndim=2,
                           flags='aligned, contiguous'),
    c_int,
    c_double,
    np.ctypeslib.ndpointer(float, ndim=2,
                           flags='aligned, contiguous,'\
                           'writeable'),
    c_int
]

if __name__ == '__main__':
    weights = np.array([[1.,2], [3,4]])
    biases = np.atleast_2d(np.array([5.,6]).T)
    act = 'tanh'
    input = np.atleast_2d(np.array([8, 9]).T)

    n=40
    weights = np.random.rand(n, n)
    biases = np.random.rand(1, n)
    input = np.random.rand(1, n)

    layer = Layer(weights, biases, act)
    out = layer.apply(input)
    print(out)
    embed()
