/*
 * MxSurfaceSimulator.cpp
 *
 *  Created on: Mar 28, 2019
 *      Author: andy
 */

#include <MxSurfaceSimulator.h>

#include <iostream>




/**
 * tp_alloc(type) to allocate storage
 * tp_new(type, args) to create blank object
 * tp_init(obj, args) to initialize object
 */

static int init(PyObject *self, PyObject *args, PyObject *kwds)
{
    std::cout << MX_FUNCTION << std::endl;
    return 0;
}

static PyObject *Noddy_name(MxSurfaceSimulator* self)
{
    return PyUnicode_FromFormat("%s %s", "foo", "bar");
}


static PyMethodDef methods[] = {
    {"name", (PyCFunction)Noddy_name, METH_NOARGS,
     "Return the name, combining the first and last name"
    },
    {NULL}  /* Sentinel */
};


static PyTypeObject SurfaceSimulatorType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    .tp_name = "mechanica.SurfaceSimulator",
    .tp_basicsize = sizeof(MxSurfaceSimulator),
    .tp_itemsize = 0,
    .tp_dealloc = 0,
    .tp_print = 0, 
    .tp_getattr = 0, 
    .tp_setattr = 0, 
    .tp_as_async = 0, 
    .tp_repr = 0, 
    .tp_as_number = 0, 
    .tp_as_sequence = 0, 
    .tp_as_mapping = 0, 
    .tp_hash = 0, 
    .tp_call = 0, 
    .tp_str = 0, 
    .tp_getattro = 0, 
    .tp_setattro = 0, 
    .tp_as_buffer = 0, 
    .tp_flags = Py_TPFLAGS_DEFAULT, 
    .tp_doc = 0, 
    .tp_traverse = 0, 
    .tp_clear = 0, 
    .tp_richcompare = 0, 
    .tp_weaklistoffset = 0, 
    .tp_iter = 0, 
    .tp_iternext = 0, 
    .tp_methods = methods, 
    .tp_members = 0, 
    .tp_getset = 0, 
    .tp_base = 0, 
    .tp_dict = 0, 
    .tp_descr_get = 0, 
    .tp_descr_set = 0, 
    .tp_dictoffset = 0, 
    .tp_init = init, 
    .tp_alloc = 0, 
    .tp_new = PyType_GenericNew, 
    .tp_free = 0,  
    .tp_is_gc = 0, 
    .tp_bases = 0, 
    .tp_mro = 0, 
    .tp_cache = 0, 
    .tp_subclasses = 0, 
    .tp_weaklist = 0, 
    .tp_del = 0, 
    .tp_version_tag = 0, 
    .tp_finalize = 0, 
};


/*
static PyTypeObject SurfaceSimulatorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "mechanica.SurfaceSimulator",
    .tp_doc = "Custom objects",
    .tp_basicsize = sizeof(MxSurfaceSimulator),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = init,
    .tp_methods = methods
};
*/

PyTypeObject *MxSurfaceSimuator_Type = &SurfaceSimulatorType;



HRESULT MxSurfaceSimulator_init(PyObject* m) {

    std::cout << MX_FUNCTION << std::endl;


    if (PyType_Ready((PyTypeObject *)MxSurfaceSimuator_Type) < 0)
        return E_FAIL;



    Py_INCREF(MxSurfaceSimuator_Type);
    PyModule_AddObject(m, "SurfaceSimulator", (PyObject *) MxSurfaceSimuator_Type);

    return 0;
}





