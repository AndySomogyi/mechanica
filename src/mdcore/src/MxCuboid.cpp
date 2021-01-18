/*
 * MxCuboid.cpp
 *
 *  Created on: Jan 17, 2021
 *      Author: andy
 */

#include <MxCuboid.hpp>
#include <MxConvert.hpp>
#include <engine.h>


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
}

static PyMethodDef cuboid_methods[] = {
    { NULL, NULL, 0, NULL }
};

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
        
        if(!SUCCEEDED((err = engine_addcuboid(&_Engine, &c, &p)))) {
            return err;
        }
        
        p->extents = {1, 1, 1};
        
        p->_handle = handle;
        
        Py_INCREF(handle);
        
        return 0;
    }
    catch (const std::exception &e) {
        return C_EXP(e);
    }
}


PyGetSetDef cuboid_getsets[] = {
    {
        .name = "position",
        .get = [](PyObject *obj, void *p) -> PyObject* {
            CUBOID_SELF(obj);
            return mx::cast(self->position);
        },
        .set = [](PyObject *obj, PyObject *val, void *p) -> int {
            try {
                CUBOID_PROP_SELF(obj);
                Magnum::Vector3 vec = mx::cast<Magnum::Vector3>(val);
                self->position = vec;
                return 0;
            }
            catch (const std::exception &e) {
                return C_EXP(e);
            }
        },
        .doc = "test doc",
        .closure = NULL
    },
    {
        .name = "velocity",
        .get = [](PyObject *obj, void *p) -> PyObject* {
            CUBOID_SELF(obj);
            return mx::cast(self->velocity);
        },
        .set = [](PyObject *obj, PyObject *val, void *p) -> int {
            try {
                int id = ((MxParticleHandle*)obj)->id;
                Magnum::Vector3 *vec = &_Engine.s.partlist[id]->velocity;
                *vec = mx::cast<Magnum::Vector3>(val);
                return 0;
            }
            catch (const std::exception &e) {
                return C_EXP(e);
            }
        },
        .doc = "test doc",
        .closure = NULL
    },
    {
        .name = "force",
        .get = [](PyObject *obj, void *p) -> PyObject* {
            int id = ((MxParticleHandle*)obj)->id;
            Magnum::Vector3 *vec = &_Engine.s.partlist[id]->force;
            return mx::cast(*vec);
        },
        .set = [](PyObject *obj, PyObject *val, void *p) -> int {
            try {
                int id = ((MxParticleHandle*)obj)->id;
                Magnum::Vector3 *vec = &_Engine.s.partlist[id]->force;
                *vec = mx::cast<Magnum::Vector3>(val);
                return 0;
            }
            catch (const std::exception &e) {
                return C_EXP(e);
            }
        },
        .doc = "test doc",
        .closure = NULL
    },
    {
        .name = "id",
        .get = [](PyObject *obj, void *p) -> PyObject* {
            CUBOID_SELF(obj)
            return carbon::cast(self->id);
        },
        .set = [](PyObject *obj, PyObject *val, void *p) -> int {
            PyErr_SetString(PyExc_ValueError, "read only property");
            return -1;
        },
        .doc = "test doc",
        .closure = NULL
    },
    {
        .name = "flags",
        .get = [](PyObject *obj, void *p) -> PyObject* {
            CUBOID_SELF(obj);
            return carbon::cast(self->flags);
        },
        .set = [](PyObject *obj, PyObject *val, void *p) -> int {
            PyErr_SetString(PyExc_ValueError, "read only property");
            return -1;
        },
        .doc = "test doc",
        .closure = NULL
    },
    {
        .name = "species",
        .get = [](PyObject *obj, void *p) -> PyObject* {
            CUBOID_SELF(obj);
            if(self->state_vector) {
                Py_INCREF(self->state_vector);
                return (PyObject*)self->state_vector;
            }
            Py_RETURN_NONE;
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
    .tp_getset =         cuboid_getsets,
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


HRESULT _MxCuboid_Init(PyObject* m) {
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
