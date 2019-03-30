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

PyTypeObject *MxSurfaceSimuator_Type = &SurfaceSimulatorType;

static PyModuleDef custommodule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "custom",
    .m_doc = "Example module that creates an extension type.",
    .m_size = -1,
};

HRESULT MxSurfaceSimulator_init(PyObject* m) {

    std::cout << MX_FUNCTION << std::endl;


    if (PyType_Ready((PyTypeObject *)MxSurfaceSimuator_Type) < 0)
        return E_FAIL;



    Py_INCREF(MxSurfaceSimuator_Type);
    PyModule_AddObject(m, "SurfaceSimulator", (PyObject *) MxSurfaceSimuator_Type);

    return 0;
}





