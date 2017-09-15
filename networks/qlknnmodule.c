
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
#include "mkl_vml_functions.h"

static PyObject *ErrorObject;

typedef struct {
    PyObject_HEAD
    PyObject            *x_attr;        /* Attributes dictionary */
    PyArrayObject       *weights;
    PyArrayObject       *biases;
    PyObject *activation;
} LayerObject;

static PyTypeObject Layer_Type;

#define LayerObject_Check(v)      (Py_TYPE(v) == &Layer_Type)

//static LayerObject *
//newLayerObject(PyObject *arg)
//{
//    LayerObject *self;
//    self = PyObject_New(LayerObject, &Layer_Type);
//    if (self == NULL)
//        return NULL;
//    self->x_attr = NULL;
//    self->weights = NULL;
//    self->biases = NULL;
//    self->activation = NULL;
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
    self->weights = NULL;
    self->biases = NULL;
    self->activation = NULL;
    return (PyObject *)self;
}


static int
Layer_init(LayerObject *self, PyObject *args, PyObject *kwds)
{
    //int *weights=NULL, *biases=NULL, *activation=NULL, *tmp;
    //PyArrayObject *weights, *arr1;
    //PyObject arg1;
    //if (! PyArg_ParseTuple(args, "Oll;", &arg1, &self->biases, &self->activation))
    //    return -1;

    //arr1 = (PyArrayObject*)PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_IN_ARRAY);
    //if (arr1 == NULL)
    //    a
    //    return -1;
    PyObject *activation=NULL, *tmp;
    PyArrayObject *weights=NULL, *biases=NULL, *tmpArray;

    if (!PyArg_ParseTuple(args, "O!O!O", &PyArray_Type, &weights, &PyArray_Type, &biases,
        &activation)) return -1;

    //arr1 = (PyArrayObject*)PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_IN_ARRAY);
    //if (arr1 == NULL) return -1;

    tmpArray = self->weights;
    Py_INCREF(weights);
    self->weights = weights;
    Py_XDECREF(tmpArray);
    //self->weights = weights;

    tmpArray = self->biases;
    Py_INCREF(biases);
    self->biases = biases;
    Py_XDECREF(tmpArray);

    //tmp = self->biases;
    //Py_INCREF(biases);
    //self->biases = biases;
    //Py_XDECREF(tmp);

    tmp = self->activation;
    Py_INCREF(activation);
    self->activation = activation;
    Py_XDECREF(tmp);

    return 0;
}

/* Layer methods */

static void
Layer_dealloc(LayerObject *self)
{
    Py_XDECREF(self->x_attr);
    Py_XDECREF(self->weights);
    Py_XDECREF(self->biases);
    Py_XDECREF(self->activation);
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

static PyArrayObject *
Layer_apply(LayerObject *self, PyObject *args)
{
    PyArrayObject *input=NULL, *out=NULL,  *tmp;
    double *A, *B, *C;
    int m, n, k, i, j;
    double alpha, beta;

    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &input, &PyArray_Type, &out)) return NULL;
//tmp = (int)self->biases + (int)self->weights + (int)self->activation;
    //PyArrayObject *weights = self->weights;
    //PyArrayObject *biases = self->biases;
    //double *  a_data_ptr =  (double *) self->weights->data;
    //double *  b_data_ptr =  (double *) self->weights->data;
    //double *  c_data_ptr =  (double *) self->weights->data;
    //double *  a_data_ptr = PyArray_DATA(self->weights);
    //m = self->weights->dimensions[0];
    //k = self->weights->dimensions[1];
    //n = self->weights->dimensions[1];
    //long *shape_input, *shape_out, *shape_weight, *shape_bias;
    //shape_input = PyArray_SHAPE(input);
    m = input->dimensions[0];
    k = input->dimensions[1];
    n = self->weights->dimensions[1];
    if (k != self->weights->dimensions[0]) {
         printf( "\n ERROR! A and B not compatible \n\n");
         return NULL;
    }
    alpha = 1.0; beta = 0.0;

    A = pyvector_to_Carrayptrs(input);
    B = pyvector_to_Carrayptrs(self->weights);
    C = pyvector_to_Carrayptrs(out);

    MKL_INT incx, incy;
    incx = 1;
    incy = 1;


    for (i = 0; i < m; i++)
    {
        cblas_dcopy(n, PyArray_DATA(self->biases), incx, C + i*n, incy);
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
    vdTanh(m*n, C, C);
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

static PyObject *
Layer_getattro(LayerObject *self, PyObject *name)
{
    if (self->x_attr != NULL) {
        PyObject *v = PyDict_GetItem(self->x_attr, name);
        if (v != NULL) {
            Py_INCREF(v);
            return v;
        }
    }
    return PyObject_GenericGetAttr((PyObject *)self, name);
}

static int
Layer_setattr(LayerObject *self, const char *name, PyObject *v)
{
    if (self->x_attr == NULL) {
        self->x_attr = PyDict_New();
        if (self->x_attr == NULL)
            return -1;
    }
    if (v == NULL) {
        int rv = PyDict_DelItemString(self->x_attr, name);
        if (rv < 0)
            PyErr_SetString(PyExc_AttributeError,
                "delete non-existing Xxo attribute");
        return rv;
    }
    else
        return PyDict_SetItemString(self->x_attr, name, v);
}
static PyMemberDef Layer_members[] = {
    {"weights", T_OBJECT_EX, offsetof(LayerObject, weights), 0,
     "first name"},
    {"biases", T_OBJECT_EX, offsetof(LayerObject, biases), 0,
     "last name"},
    {"activation", T_OBJECT_EX, offsetof(LayerObject, activation), 0,
     "noddy number"},
    {NULL}  /* Sentinel */
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
    (getattrfunc)0,             /*tp_getattr*/
    (setattrfunc)Layer_setattr,   /*tp_setattr*/
    0,                          /*tp_reserved*/
    0,                          /*tp_repr*/
    0,                          /*tp_as_number*/
    0,                          /*tp_as_sequence*/
    0,                          /*tp_as_mapping*/
    0,                          /*tp_hash*/
    0,                          /*tp_call*/
    0,                          /*tp_str*/
    (getattrofunc)Layer_getattro, /*tp_getattro*/
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
    0,                          /*tp_getset*/
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
