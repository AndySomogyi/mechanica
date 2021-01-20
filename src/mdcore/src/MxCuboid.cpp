/*
 * MxCuboid.cpp
 *
 *  Created on: Jan 17, 2021
 *      Author: andy
 */

#include <MxCuboid.hpp>
#include <MxConvert.hpp>
#include <Magnum/Math/Matrix4.h>
#include <engine.h>
#include <cuboid_eval.hpp>

#define CUBOID_SELF(handle) \
    MxCuboid *self = &_Engine.s.cuboids[((MxCuboidHandle*)handle)->id]; \
    if(self == NULL) { \
        PyErr_SetString(PyExc_ReferenceError, "Cuboid has been destroyed or is invalid"); \
        return NULL; \
    }

#define CUBOID_PROP_SELF(handle) \
    MxCuboid *self = &_Engine.s.cuboids[((MxCuboidHandle*)handle)->id]; \
    if(self == NULL) { \
        PyErr_SetString(PyExc_ReferenceError, "Cuboid has been destroyed or is invalid"); \
        return -1; \
    }


MxCuboid::MxCuboid() {
    bzero(this, sizeof(MxCuboid));
    orientation = Magnum::Quaternion();
}


static PyObject* cuboid_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
    //std::cout << MX_FUNCTION << ", type: " << type->tp_name << std::endl;
    return PyType_GenericNew(type, args, kwargs);
}


int cuboid_init(MxCuboidHandle *handle, PyObject *args, PyObject *kwds) {
    
    try {
        MxCuboid c;
        
        HRESULT err;
        
        MxCuboid *p;
        
        c.position = mx::arg<Magnum::Vector3>("pos", 0, args, kwds, engine_center());
        c.size = mx::arg<Magnum::Vector3>("size", 1, args, kwds, Magnum::Vector3{1, 1, 1});
        
        Magnum::Vector3 angle = mx::arg<Magnum::Vector3>("orientation", 2, args, kwds, Magnum::Vector3{0, 0, 0});
        
        Magnum::Quaternion qx = Magnum::Quaternion::rotation(Magnum::Rad(angle[0]), Magnum::Vector3::xAxis());
        Magnum::Quaternion qy = Magnum::Quaternion::rotation(Magnum::Rad(angle[1]), Magnum::Vector3::yAxis());
        Magnum::Quaternion qz = Magnum::Quaternion::rotation(Magnum::Rad(angle[2]), Magnum::Vector3::zAxis());
        
        c.orientation = qx * qy * qz;
        
        if(!SUCCEEDED((err = engine_addcuboid(&_Engine, &c, &p)))) {
            return err;
        }
        
        p->_handle = handle;
        
        Py_INCREF(handle);
        
        return 0;
    }
    catch (const std::exception &e) {
        return C_EXP(e);
    }
}

static PyObject* cuboid_scale(MxCuboidHandle *_self, PyObject *args, PyObject *kwargs) {
    try {
        CUBOID_SELF(_self);
        
        Magnum::Vector3 scale = mx::arg<Magnum::Vector3>("scale", 0, args, kwargs);
        
        self->size = Magnum::Matrix4::scaling(scale).transformVector(self->size);
        
        Py_RETURN_NONE;
    }
    catch(const std::exception &e) {
        C_RETURN_EXP(e);
    }
}

static PyMethodDef cuboid_methods[] = {
    { "scale", (PyCFunction)cuboid_scale, METH_VARARGS | METH_KEYWORDS, NULL },
    { NULL, NULL, 0, NULL }
};


PyGetSetDef cuboid_getsets[] = {
    {
        .name = "size",
        .get = [](PyObject *obj, void *p) -> PyObject* {
            CUBOID_SELF(obj);
            return mx::cast(self->size);
        },
        .set = [](PyObject *obj, PyObject *val, void *p) -> int {
            PyErr_SetString(PyExc_PermissionError, "read only");
            return -1;
        },
        .doc = "test doc",
        .closure = NULL
    },
    {NULL}
};

PyTypeObject MxCuboid_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name =           "Cuboid",
    .tp_basicsize =      sizeof(MxCuboidHandle),
    .tp_itemsize =       0,
    .tp_dealloc =        0,
                         0, // .tp_print changed to tp_vectorcall_offset in python 3.8
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
    .tp_methods =        cuboid_methods,
    .tp_members =        0,
    .tp_getset =         0,
    .tp_base =           0,
    .tp_dict =           0,
    .tp_descr_get =      0,
    .tp_descr_set =      0,
    .tp_dictoffset =     0,
    .tp_init =           (initproc)cuboid_init,
    .tp_alloc =          0,
    .tp_new =            cuboid_new,
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


int MxCuboid_Check(PyObject *obj) {
    if(obj) {
        return PyObject_IsInstance(obj, (PyObject*)&MxCuboid_Type);
    }
    return 0;
}

/**
 * check if a object is a cuboid type
 */
int MxCuboidType_Check(PyObject *obj) {
    if(obj && PyType_Check(obj)) {
        return PyObject_IsSubclass(obj, (PyObject*)&MxCuboid_Type);
    }
    return 0;
}


HRESULT _MxCuboid_Init(PyObject* m) {
    
    // WARNING: make sure MxBody is initialized before cuboid.
    MxCuboid_Type.tp_base = &MxBody_Type;
    
    if (PyType_Ready((PyTypeObject*)&MxCuboid_Type) < 0) {
        return E_FAIL;
    }

    Py_INCREF(&MxCuboid_Type);
    if (PyModule_AddObject(m, "Cuboid", (PyObject *)&MxCuboid_Type) < 0) {
        Py_DECREF(&MxCuboid_Type);
        return E_FAIL;
    }

    return S_OK;
}


