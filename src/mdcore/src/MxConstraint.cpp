/*
 * MxConstraint.cpp
 *
 *  Created on: Aug 26, 2020
 *      Author: andy
 */

#include <MxConstraint.hpp>
#include <iostream>


/**
 * force type
 *
 * set tp_basicsize to 1 (bytes), so calls to alloc can add size of additional stuff
 * in bytes.
 *
 * force type is sort of meant to be a metatype, in that we can have lots of
 * different instances of force functions, that have different attributes, but
 * only want to have one type.
 */
PyTypeObject MxConstraint_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name =           "Constraint",
    .tp_basicsize =      sizeof(MxConstraint),
    .tp_itemsize =       1,
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
    .tp_methods =        0,
    .tp_members =        0,
    .tp_getset =         0,
    .tp_base =           0,
    .tp_dict =           0,
    .tp_descr_get =      (descrgetfunc)0,
    .tp_descr_set =      0,
    .tp_dictoffset =     0,
    .tp_init =           (initproc)0,
    .tp_alloc =          0,
    .tp_new =            0,
    .tp_free =           [] (void* p) {
        std::cout << "freeing force" << std::endl;
        PyObject_Free(p);
    },
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

HRESULT _MxConstraint_init(PyObject *m) {
    std::cout << MX_FUNCTION << std::endl;
    return S_OK;
}


