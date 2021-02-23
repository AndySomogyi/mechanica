/*
 * CKeyEvent.cpp
 *
 *  Created on: Dec 29, 2020
 *      Author: andy
 */

#include "MxKeyEvent.hpp"
#include <MxConvert.hpp>
#include <CConvert.hpp>

#include <iostream>

struct MxKeyEvent : CEvent
{
    Magnum::Platform::GlfwApplication::KeyEvent *glfw_event;
};

static PyObject *delegate = NULL;


PyGetSetDef keyevent_getsets[] = {
    {
        .name = "key_name",
        .get = [](PyObject *obj, void *p) -> PyObject* {
            MxKeyEvent *e = (MxKeyEvent*)(obj);
            return carbon::cast(e->glfw_event->keyName());
        },
        .set = [](PyObject *obj, PyObject *val, void *p) -> int {
            PyErr_SetString(PyExc_Exception, "key_name is read-only");
            return -1;
        },
        .doc = "test doc",
        .closure = NULL
    },
    {NULL}
};

HRESULT MxKeyEvent_Invoke(Magnum::Platform::GlfwApplication::KeyEvent &ke)
{
    if(delegate) {
        MxKeyEvent *event = (MxKeyEvent*)PyType_GenericAlloc(&MxKeyEvent_Type, 0);
        
        event->glfw_event = &ke;
        
        PyObject *args = PyTuple_New(1);
        PyTuple_SET_ITEM(args, 0, event);
        
        PyObject *result = PyObject_Call(delegate, args, NULL);
        
        if(result) {
            Py_DECREF(result);
        }
        else {
            PyErr_Print();
            PyErr_Clear();
        }
        
        Py_DECREF(args);
    }

    
    // TODO: check result code
    return S_OK;
}

HRESULT MxKeyEvent_Add(MxKeyEvent *e)
{

    return S_OK;
}

MxKeyEvent *MxKeyEvent_New(PyObject *args, PyObject *kwargs) {
    return NULL;
}



PyObject* MxKeyEvent_AddDelegate(PyObject *module, PyObject *args, PyObject *kwargs)
{
    Log(LOG_DEBUG) << "obj: " << carbon::str(module)
                   << "args: " << carbon::str(args)
                   << "kwargs: " << carbon::str(kwargs);
    
    PyObject *method = NULL;
    
    if(args && PyTuple_GET_SIZE(args) > 0) {
        method = PyTuple_GET_ITEM(args, 0);
    }
    
    if(PyCallable_Check(method)) {
        delegate = method;
        Py_INCREF(method);
    }
    
    Py_RETURN_NONE;
}



PyTypeObject MxKeyEvent_Type = {
    CVarObject_HEAD_INIT(NULL, 0)
    "KeyEvent"                         , // .tp_name
    sizeof(MxKeyEvent)                  , // .tp_basicsize
    0                                     , // .tp_itemsize
    0       , // .tp_dealloc
    0                                     , // .tp_print
    0                                     , // .tp_getattr
    0                                     , // .tp_setattr
    0                                     , // .tp_as_async
    0             , // .tp_repr
    0                                     , // .tp_as_number
    0                                     , // .tp_as_sequence
    0                  , // .tp_as_mapping
    0                                     , // .tp_hash
    0                                     , // .tp_call
    (reprfunc)0             , // .tp_str
    (getattrofunc)0    , // .tp_getattro                                     , // .tp_getattro
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
    0                   , // .tp_methods
    0                                     , // .tp_members
    keyevent_getsets                                     , // .tp_getset
    0                                     , // .tp_base
    0                                     , // .tp_dict
    0                                     , // .tp_descr_get
    0                                     , // .tp_descr_set
    0                                     , // .tp_dictoffset
    (initproc)0            , // .tp_init
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
    0                                     , // .tp_next =
#endif
};



HRESULT _MxKeyEvent_Init(PyObject* m) {
    
    if (PyType_Ready((PyTypeObject*)&MxKeyEvent_Type) < 0) {
        return E_FAIL;
    }
    
    Py_INCREF(&MxKeyEvent_Type);
    if (PyModule_AddObject(m, "KeyEvent", (PyObject *)&MxKeyEvent_Type) < 0) {
        Py_DECREF(&MxKeyEvent_Type);
        return E_FAIL;
    }
    

    
    return S_OK;
}


