
/* Use this file as a template to start implementing a module that
   also declares object types. All occurrences of 'Xxo' should be changed
   to something reasonable for your objects. After that, all other
   occurrences of 'xx' should be changed to something reasonable for your
   module. If your module is named foo your sourcefile should be named
   foomodule.c.

   You will probably want to delete all references to 'x_attr' and add
   your own types of attributes instead.  Maybe you want to name your
   local variables other than 'self'.  If your object type is needed in
   other files, you'll have to create a file "foobarobject.h"; see
   floatobject.h for an example. */

/* Xxo objects */

#include "Python.h"
#include "structmember.h"
#include "numpy/ndarraytypes.h"
#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"
#include "mkl.h"
#include "omp.h"
#include "mkl_vml_functions.h"
#include "mkl_vml.h"

static PyObject *ErrorObject;

enum act {
    TANH,
    RELU,
    NONE,
    ERR
};

typedef struct {
    PyObject_HEAD
    PyObject            *x_attr;        /* Attributes dictionary */
    PyArrayObject       *_weights;
    PyArrayObject       *_biases;
    enum act _activation;
} LayerObject;


static PyTypeObject Layer_Type;

#define LayerObject_Check(v)      (Py_TYPE(v) == &Layer_Type)
#define PyArray_ISOWNDATA(m) PyArray_CHKFLAGS(m, NPY_ARRAY_OWNDATA)

//static LayerObject *
//newLayerObject(PyObject *arg)
//{
//    LayerObject *self;
//    self = PyObject_New(LayerObject, &Layer_Type);
//    if (self == NULL)
//        return NULL;
//    self->x_attr = NULL;
//    self->_weights = NULL;
//    self->_biases = NULL;
//    self->_activation = NULL;
//    return self;
//}

static PyObject *
Layer_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    LayerObject *self;
    //self = PyObject_New(LayerObject, &Layer_Type);
    self = (LayerObject *)type->tp_alloc(type, 0);
    if (self == NULL)
        return NULL;
    self->x_attr = NULL;
    self->_weights = NULL;
    self->_biases = NULL;
    self->_activation = ERR;
    return (PyObject *)self;
}


static int
Layer_init(LayerObject *self, PyObject *args, PyObject *kwds)
{
    //int *_weights=NULL, *_biases=NULL, *_activation=NULL, *tmp;
    //PyArrayObject *_weights, *arr1;
    //PyObject arg1;
    //if (! PyArg_ParseTuple(args, "Oll;", &arg1, &self->_biases, &self->_activation))
    //    return -1;

    //arr1 = (PyArrayObject*)PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_IN_ARRAY);
    //if (arr1 == NULL)
    //    a
    //    return -1;
    PyObject *tmp;
    PyArrayObject *_weights=NULL, *_biases=NULL, *tmpArray;
    char *_activation;

    if (!PyArg_ParseTuple(args, "O!O!s", &PyArray_Type, &_weights, &PyArray_Type, &_biases,
        &_activation)) return -1;

    //arr1 = (PyArrayObject*)PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_IN_ARRAY);
    //if (arr1 == NULL) return -1;

    tmpArray = self->_weights;
    Py_INCREF(_weights);
    self->_weights = _weights;
    Py_XDECREF(tmpArray);
    //self->_weights = _weights;

    tmpArray = self->_biases;
    Py_INCREF(_biases);
    self->_biases = _biases;
    Py_XDECREF(tmpArray);

    //tmp = self->_biases;
    //Py_INCREF(_biases);
    //self->_biases = _biases;
    //Py_XDECREF(tmp);

    if (!strcmp(_activation, "tanh"))
        self->_activation = TANH;
    else if (!strcmp(_activation, "relu"))
        self->_activation = RELU;
    else if (!strcmp(_activation, "none"))
        self->_activation = NONE;
    else
        return -1;
    //tmp = self->_activation;
    //Py_INCREF(_activation);
    //self->_activation = _activation;
    //Py_XDECREF(tmp);

    return 0;
}

/* Layer methods */

static void
Layer_dealloc(LayerObject *self)
{
    Py_XDECREF(self->x_attr);
    Py_XDECREF(self->_weights);
    Py_XDECREF(self->_biases);
    //Py_XDECREF(self->_activation);
    //PyObject_Del(self);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject *
Layer_demo(LayerObject *self, PyObject *args)
{
    if (!PyArg_ParseTuple(args, ":demo"))
        return NULL;
    Py_INCREF(Py_None);
    return Py_None;
}

double *pyvector_to_Carrayptrs(PyArrayObject *arrayin)  {
    int i,n;

    n=arrayin->dimensions[0];
    return (double *) arrayin->data;  /* pointer to arrayin data as double */
}

int array_is_safe(PyArrayObject *arrayin) {
    if (!(PyArray_ISOWNDATA(arrayin) && PyArray_ISCARRAY(arrayin)))
    {
        printf("ERROR: Array is unsafe! Is it contagious C-ordered?\n");
        return 0;
    }
    return 1;
}

static PyArrayObject *
Layer_apply(LayerObject *self, PyObject *args)
{
    PyArrayObject *input=NULL, *out=NULL,  *tmp;
    double *A, *B, *C;
    int m, n, k, i, j;
    double alpha, beta;

    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &input, &PyArray_Type, &out)) return NULL;

    if (!array_is_safe(input)) return NULL;
    if (!array_is_safe(out)) return NULL;
    mkl_set_num_threads(1);
    omp_set_num_threads(1);

    m = input->dimensions[0];
    k = input->dimensions[1];
    n = self->_weights->dimensions[1];
    if (k != self->_weights->dimensions[0]) {
         printf( "\n ERROR! A and B not compatible \n\n");
         return NULL;
    }
    alpha = 1.0; beta = 1.0;

    A = pyvector_to_Carrayptrs(input);
    B = pyvector_to_Carrayptrs(self->_weights);
    C = pyvector_to_Carrayptrs(out);

    MKL_INT incx, incy;
    incx = 1;
    incy = 1;
    for (i = 0; i < m; i++)
    {
        cblas_dcopy(n, PyArray_DATA(self->_biases), incx, C + i*n, incy);
    }

    if (A == NULL || B == NULL || C == NULL) {
      printf( "\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
      mkl_free(A);
      mkl_free(B);
      mkl_free(C);
      return NULL;
    }


    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, alpha, A, k, B, n, beta, C, n);
    //mode = VML_HA;
    //vmlSetMode(VML_LA);
    double *zeros;
    switch(self->_activation) {
        case TANH:
            vdTanh(m*n, C, C);
            break;
        case NONE:
            break;
        case RELU:
            zeros = (double *)calloc( m*n, sizeof( double ) );
            for (int ii = 0; ii < m*n; ii++)
                zeros[ii] = 0.0;
            vdFmax(m*n, C, zeros, C);
            free(zeros);
            break;
        default:
            PyErr_SetString(PyExc_AttributeError, "_activation has an unknown value");
            return NULL;
    }
    Py_INCREF(out);
    return out;
}

static PyMethodDef Layer_methods[] = {
    {"demo",            (PyCFunction)Layer_demo,  METH_VARARGS,
        PyDoc_STR("demo() -> None")},
    {"apply",            (PyCFunction)Layer_apply,  METH_VARARGS,
        PyDoc_STR("demo() -> None")},
    {NULL,              NULL}           /* sentinel */
};

//static PyObject *
//Layer_getattro(LayerObject *self, PyObject *name)
//{
//    if (self->x_attr != NULL) {
//        PyObject *v = PyDict_GetItem(self->x_attr, name);
//        if (v != NULL) {
//            Py_INCREF(v);
//            return v;
//        }
//    }
//    return PyObject_GenericGetAttr((PyObject *)self, name);
//}

//static int
//Layer_setattr(LayerObject *self, const char *name, PyObject *v)
//{
//    if (self->x_attr == NULL) {
//        self->x_attr = PyDict_New();
//        if (self->x_attr == NULL)
//            return -1;
//    }
//    if (v == NULL) {
//        int rv = PyDict_DelItemString(self->x_attr, name);
//        if (rv < 0)
//            PyErr_SetString(PyExc_AttributeError,
//                "delete non-existing Xxo attribute");
//        return rv;
//    }
//    else
//        return PyDict_SetItemString(self->x_attr, name, v);
//}
static PyMemberDef Layer_members[] = {
    {"_weights", T_OBJECT_EX, offsetof(LayerObject, _weights), 0,
     "weights"},
    {"_biases", T_OBJECT_EX, offsetof(LayerObject, _biases), 0,
     "biases"},
    {NULL}  /* Sentinel */
};

static PyObject *
Layer_get_activation(LayerObject *self, void *closure)
{
    PyObject *output;
    switch(self->_activation) {
        case TANH:
            output = PyUnicode_FromString("tanh");
            break;
        case RELU:
            output = PyUnicode_FromString("relu");
            break;
        case NONE:
            output = PyUnicode_FromString("none");
            break;
        default:
            PyErr_SetString(PyExc_AttributeError, "_activation has an unknown value");
            output = NULL;
    }

    return output;
}

static int
Layer_set_activation(LayerObject *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the _activation attribute");
        return -1;
    }

    if (! PyUnicode_Check(value)) {
        PyErr_SetString(PyExc_TypeError,
                        "_activation attribute value must be a string");
        return -1;
    }

    if (!PyUnicode_CompareWithASCIIString(value, "tanh"))
        self->_activation = TANH;
    else if (!PyUnicode_CompareWithASCIIString(value, "relu"))
        self->_activation = RELU;
    else if (!PyUnicode_CompareWithASCIIString(value, "none"))
        self->_activation = NONE;
    else
    {
        PyErr_SetString(PyExc_ValueError, "Unkown activation function, cannot set");
        return -1;
    }

    return 0;
}

static PyGetSetDef Layer_getseters[] = {
    {"_activation",
        (getter)Layer_get_activation,
        (setter)Layer_set_activation,
        "activation function", NULL},
    {NULL}
};

static PyTypeObject Layer_Type = {
    /* The ob_type field must be initialized in the module init function
     * to be portable to Windows without using C++. */
    PyVarObject_HEAD_INIT(NULL, 0)
    "qlknnmodule.Layer",             /*tp_name*/
    sizeof(LayerObject),          /*tp_basicsize*/
    0,                          /*tp_itemsize*/
    /* methods */
    (destructor)Layer_dealloc,    /*tp_dealloc*/
    0,                          /*tp_print*/
    0,             /*tp_getattr*/
    //(setattrfunc)Layer_setattr,   /*tp_setattr*/
    0,
    0,                          /*tp_reserved*/
    0,                          /*tp_repr*/
    0,                          /*tp_as_number*/
    0,                          /*tp_as_sequence*/
    0,                          /*tp_as_mapping*/
    0,                          /*tp_hash*/
    0,                          /*tp_call*/
    0,                          /*tp_str*/
    0,                          /*tp_getattro*/
    0,                          /*tp_setattro*/
    0,                          /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT,         /*tp_flags*/
    0,                          /*tp_doc*/
    0,                          /*tp_traverse*/
    0,                          /*tp_clear*/
    0,                          /*tp_richcompare*/
    0,                          /*tp_weaklistoffset*/
    0,                          /*tp_iter*/
    0,                          /*tp_iternext*/
    Layer_methods,                /*tp_methods*/
    Layer_members,                          /*tp_members*/
    Layer_getseters,                          /*tp_getset*/
    0,                          /*tp_base*/
    0,                          /*tp_dict*/
    0,                          /*tp_descr_get*/
    0,                          /*tp_descr_set*/
    0,                          /*tp_dictoffset*/
    (initproc)Layer_init,                 /*tp_init*/
    0,                          /*tp_alloc*/
    Layer_new,                  /*tp_new*/
    0,                          /*tp_free*/
    0,                          /*tp_is_gc*/
};
/* --------------------------------------------------------------------- */

/* Function of two integers returning integer */

PyDoc_STRVAR(qlknn_foo_doc,
"foo(i,j)\n\
\n\
Return the sum of i and j.");

static PyObject *
qlknn_foo(PyObject *self, PyObject *args)
{
    long i, j;
    long res;
    if (!PyArg_ParseTuple(args, "ll:foo", &i, &j))
        return NULL;
    res = i+j; /* XXX Do something here */
    return PyLong_FromLong(res);
}

/* List of functions defined in the module */

static PyMethodDef qlknn_methods[] = {
//    {"roj",             xx_roj,         METH_VARARGS,
//        PyDoc_STR("roj(a,b) -> None")},
    {"foo",             qlknn_foo,         METH_VARARGS,
        qlknn_foo_doc},
    //{"new",             layer_new,         METH_VARARGS,
    //    PyDoc_STR("new() -> new Xx object")},
//    {"bug",             xx_bug,         METH_VARARGS,
//        PyDoc_STR("bug(o) -> None")},
    {NULL,              NULL}           /* sentinel */
};

PyDoc_STRVAR(module_doc,
"This is a template module just for instruction.");


//static int
//qlknn_exec(PyObject *m)
//{
//    /* Due to cross platform compiler issues the slots must be filled
//     * here. It's required for portability to Windows without requiring
//     * C++. */
//    //Null_Type.tp_base = &PyBaseObject_Type;
//    //Null_Type.tp_new = PyType_GenericNew;
//    //Str_Type.tp_base = &PyUnicode_Type;
//
//    /* Finalize the type object including setting type of the new type
//     * object; doing it here is required for portability, too. */
//    if (PyType_Ready(&Layer_Type) < 0)
//        goto fail;
//    PyModule_AddObject(m, "Layer", (PyObject *)&Layer_Type);
//
//    /* Add some symbolic constants to the module */
//    if (ErrorObject == NULL) {
//        ErrorObject = PyErr_NewException("qlknn.error", NULL, NULL);
//        if (ErrorObject == NULL)
//            goto fail;
//    }
//    Py_INCREF(ErrorObject);
//    PyModule_AddObject(m, "error", ErrorObject);
//
//    ///* Add Str */
//    //if (PyType_Ready(&Str_Type) < 0)
//    //    goto fail;
//    //PyModule_AddObject(m, "Str", (PyObject *)&Str_Type);
//
//    ///* Add Null */
//    //if (PyType_Ready(&Null_Type) < 0)
//    //    goto fail;
//    //PyModule_AddObject(m, "Null", (PyObject *)&Null_Type);
//    return 0;
// fail:
//    Py_XDECREF(m);
//    return -1;
//}

//static struct PyModuleDef_Slot qlknn_slots[] = {
//    {Py_mod_exec, qlknn_exec},
//    {0, NULL},
//};

static struct PyModuleDef qlknnmodule = {
    PyModuleDef_HEAD_INIT,
    "qlknn",
    module_doc,
    0,
    qlknn_methods,
    NULL,
    NULL,
    NULL,
    NULL
};

/* Export function for the module (*must* be called PyInit_xx) */

PyMODINIT_FUNC
PyInit_qlknn(void)
{
    PyObject* m;
    if (PyType_Ready(&Layer_Type) < 0)
        return NULL;

    m = PyModule_Create(&qlknnmodule);
    if (m == NULL)
        return NULL;

    Py_INCREF(&Layer_Type);
    PyModule_AddObject(m, "Layer", (PyObject *)&Layer_Type);

    /* Add some symbolic constants to the module */
    if (ErrorObject == NULL) {
        ErrorObject = PyErr_NewException("qlknn.error", NULL, NULL);
        if (ErrorObject == NULL)
            return NULL;
    }
    Py_INCREF(ErrorObject);
    PyModule_AddObject(m, "error", ErrorObject);

    ///* Add Str */
    //if (PyType_Ready(&Str_Type) < 0)
    //    goto fail;
    //PyModule_AddObject(m, "Str", (PyObject *)&Str_Type);

    ///* Add Null */
    //if (PyType_Ready(&Null_Type) < 0)
    //    goto fail;
    //PyModule_AddObject(m, "Null", (PyObject *)&Null_Type);
    import_array();
    import_ufunc();
    return m;
}
