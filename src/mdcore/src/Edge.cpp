/*
 * Edge.cpp
 *
 *  Created on: Dec 10, 2020
 *      Author: andy
 */

#include <Edge.hpp>

static PyMethodDef edge_methods[] = {
    { NULL, NULL, 0, NULL }
};

PyTypeObject Edge_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name =           "Edge",
    .tp_basicsize =      sizeof(EdgeHandle),
    .tp_itemsize =       0,
    .tp_dealloc =        0,
    .tp_print =          0,
    .tp_getattr =        0,
    .tp_setattr =        0,
    .tp_as_async =       0,
    .tp_repr =           0,
    .tp_as_number =      0,
    .tp_as_sequence =    0,
    .tp_as_mapping =     0,
    .tp_hash =           0,
    .tp_call =           0,
    .tp_str =            0,
    .tp_getattro =       0,
    .tp_setattro =       0,
    .tp_as_buffer =      0,
    .tp_flags =          Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc =            "Custom objects",
    .tp_traverse =       0,
    .tp_clear =          0,
    .tp_richcompare =    0,
    .tp_weaklistoffset = 0,
    .tp_iter =           0,
    .tp_iternext =       0,
    .tp_methods =        edge_methods,
    .tp_members =        0,
    .tp_getset =         0,
    .tp_base =           0,
    .tp_dict =           0,
    .tp_descr_get =      0,
    .tp_descr_set =      0,
    .tp_dictoffset =     0,
    .tp_init =           0,
    .tp_alloc =          0,
    .tp_new =            0,
    .tp_free =           0,
    .tp_is_gc =          0,
    .tp_bases =          0,
    .tp_mro =            0,
    .tp_cache =          0,
    .tp_subclasses =     0,
    .tp_weaklist =       0,
    .tp_del =            0,
    .tp_version_tag =    0,
    .tp_finalize =       0,
};

HRESULT _edge_init(PyObject *m) {
    if (PyType_Ready((PyTypeObject*)&Edge_Type) < 0) {
        return E_FAIL;
    }
    
    Py_INCREF(&Edge_Type);
    if (PyModule_AddObject(m, "Edge", (PyObject *)&Edge_Type) < 0) {
        Py_DECREF(&Edge_Type);
        return E_FAIL;
    }
    
    return S_OK;
}



