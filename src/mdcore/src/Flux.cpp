/*
 * Flux.cpp
 *
 *  Created on: Dec 21, 2020
 *      Author: andy
 */

#include <Flux.hpp>
#include <MxConvert.hpp>



PyTypeObject MxFluxes_Type = {
    CVarObject_HEAD_INIT(NULL, 0)
    "Flux"                                , // .tp_name
    sizeof(MxFluxes)                        , // .tp_basicsize
    sizeof(int32_t)                       , // .tp_itemsize
    (destructor )0         , // .tp_dealloc
    0                                     , // .tp_print
    0                                     , // .tp_getattr
    0                                     , // .tp_setattr
    0                                     , // .tp_as_async
    (reprfunc)0                 , // .tp_repr
    0                                     , // .tp_as_number
    0                                     , // .tp_as_sequence
    0                                     , // .tp_as_mapping
    0                                     , // .tp_hash
    0                                     , // .tp_call
    (reprfunc)0                 , // .tp_str
    0                                     , // .tp_getattro
    0                                     , // .tp_setattro
    0                                     , // .tp_as_buffer
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE , // .tp_flags
    0                                     , // .tp_doc
    0                                     , // .tp_traverse
    0                                     , // .tp_clear
    0                                     , // .tp_richcompare
    0                                     , // .tp_weaklistoffset
    0                                     , // .tp_iter
    0                                     , // .tp_iternext
    0                       , // .tp_methods
    0                                     , // .tp_members
    0                      , // .tp_getset
    0                                     , // .tp_base
    0                                     , // .tp_dict
    0                                     , // .tp_descr_get
    0                                     , // .tp_descr_set
    0                                     , // .tp_dictoffset
    (initproc)0               , // .tp_init
    0                                     , // .tp_alloc
    PyType_GenericNew                     , // .tp_new
    0                                     , // .tp_free
    0                                     , // .tp_is_gc
    0                                     , // .tp_bases
    0                                     , // .tp_mro
    0                                     , // .tp_cache
    0                                     , // .tp_subclasses
    0                                     , // .tp_weaklist
    0                                     , // .tp_del
    0                                     , // .tp_version_tag
    0                                     , // .tp_finalize
#ifdef COUNT_ALLOCS
    0                                     , // .tp_allocs
    0                                     , // .tp_frees
    0                                     , // .tp_maxalloc
    0                                     , // .tp_prev
    0                                     , // .tp_next
#endif
};



MX_BASIC_PYTHON_TYPE_INIT(Fluxes)

