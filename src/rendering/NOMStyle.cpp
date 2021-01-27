/*
 * NOMStyle.cpp
 *
 *  Created on: Jul 29, 2020
 *      Author: andy
 */

#include <rendering/NOMStyle.hpp>
#include <engine.h>
#include <space.h>
#include <pybind11/pybind11.h>
#include <MxUtil.h>
#include <MxConvert.hpp>
#include <CConvert.hpp>
#include "MxColorMapper.hpp"

static int style_init(PyObject *, PyObject *, PyObject *);

HRESULT NOMStyle_SetColor(NOMStyle *s, PyObject *o) {
    
    if(PyUnicode_Check(o)) {
        const char* str = PyUnicode_AsUTF8(o);
        s->color = Color3_Parse(str);
    }
    
    return S_OK;
}

HRESULT NOMStyle_SetFlag(NOMStyle *s, StyleFlags flag, bool value) {
    if(flag == STYLE_VISIBLE) {
        if(value) {
            s->flags |= STYLE_VISIBLE;
        }
        else {
            s->flags &= ~STYLE_VISIBLE;
        }
        return space_update_style(&_Engine.s);
    }
    return c_error(E_FAIL, "invalid flag id");
}

NOMStyle* NOMStyle_New(PyObject *args, PyObject *kwargs) {
    NOMStyle *style = (NOMStyle*)PyType_GenericNew(&NOMStyle_Type, NULL, NULL);
    if(style_init(style, args, kwargs) != 0) {
        Py_DECREF(style);
        return NULL;
    }
    return style;
}

NOMStyle* NOMStyle_NewEx(const Magnum::Color3& color, uint32_t flags) {
    NOMStyle *style = (NOMStyle*)PyType_GenericNew(&NOMStyle_Type, NULL, NULL);
    style->color = color;
    style->flags = flags;
    return style;
}

CAPI_FUNC(NOMStyle*) NOMStyle_Clone(NOMStyle* s) {
    NOMStyle *style = (NOMStyle*)PyType_GenericNew(&NOMStyle_Type, NULL, NULL);
    style->color = s->color;
    style->flags = s->flags;
    return style;
}

static PyGetSetDef getset[] = {
    {
        .name = "color",
        .get = [](PyObject *_obj, void *p) -> PyObject* {
            NOMStyle *self = (NOMStyle*)_obj;
            return pybind11::cast(self->color).release().ptr();
        },
        .set = [](PyObject *self, PyObject *val, void *p) -> int {
            return NOMStyle_SetColor((NOMStyle*)self, val);
        },
        .doc = "test doc",
        .closure = NULL
    },
    {
        .name = "visible",
        .get = [](PyObject *_obj, void *p) -> PyObject* {
            NOMStyle *self = (NOMStyle*)_obj;
            bool visible = self->flags & STYLE_VISIBLE;
            return mx::cast(visible);
        },
        .set = [](PyObject *self, PyObject *val, void *p) -> int {
            try {
                bool visible = mx::cast<bool>(val);
                return NOMStyle_SetFlag((NOMStyle*)self, STYLE_VISIBLE, visible);
            }
            catch(const std::exception &e) {
                return C_EXP(e);
            }
        },
        .doc = "test doc",
        .closure = NULL
    },
    {
        .name = "colormap",
        .get = [](PyObject *_obj, void *p) -> PyObject* {
            NOMStyle *self = (NOMStyle*)_obj;
            if(self->mapper) {
                Py_INCREF(self->mapper);
                return self->mapper;
            }
            Py_RETURN_NONE;
        },
        .set = [](PyObject *_self, PyObject *val, void *p) -> int {
            try {
                NOMStyle *self = (NOMStyle*)_self;
                if(self->mapper) {
                    if(carbon::check<std::string>(val)) {
                        std::string s = carbon::cast<std::string>(val);
                        self->mapper->set_colormap(s);
                        self->mapper_func = self->mapper->map;
                    }
                }
            }
            catch(const std::exception &e) {
                return C_EXP(e);
            }
            return 0;
        },
        .doc = "test doc",
        .closure = NULL
    },
    {NULL}
};


/**
 * particle type metatype
 */
PyTypeObject NOMStyle_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name =           "Style",
    .tp_basicsize =      sizeof(NOMStyle),
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
    .tp_methods =        0,
    .tp_members =        0,
    .tp_getset =         getset,
    .tp_base =           0,
    .tp_dict =           0,
    .tp_descr_get =      (descrgetfunc)0,
    .tp_descr_set =      0,
    .tp_dictoffset =     0,
    .tp_init =           style_init,
    .tp_alloc =          0,
    .tp_new =            PyType_GenericNew,
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



static int style_init(PyObject *_s, PyObject *args, PyObject *kwargs) {
    NOMStyle *self = (NOMStyle*)_s;
    
    
    
    if(kwargs) {
        PyObject *color = PyDict_GetItemString(kwargs, "color");
        if(color) {
            if(FAILED(NOMStyle_SetColor(self, color))) {
                return -1;
            }
        }
        else {
            // TODO default color
            self->color = Color3_Parse("steelblue");
        }
        
        PyObject *visible = PyDict_GetItemString(kwargs, "visible");
        if(visible == Py_False) {
            self->flags &= ~STYLE_VISIBLE;
        }
        else {
            self->flags |= STYLE_VISIBLE;
        }
        
        PyObject *cmap = PyDict_GetItemString(kwargs, "colormap");
        if(cmap) {
            self->mapper = MxColorMapper_New(args, cmap);
            if(self->mapper) {
                self->mapper_func = self->mapper->map;
            }
            
        }
    }
    else {
        // TODO default color
        self->color = Color3_Parse("steelblue");
        self->flags = STYLE_VISIBLE;
    }
    
    return 0;
}

int NOMStyle_Check(const PyObject *o) {
    return o && PyObject_IsInstance(const_cast<PyObject*>(o), (PyObject*)&NOMStyle_Type);
}



HRESULT _NOMStyle_init(PyObject *m)
{
    if (PyType_Ready((PyTypeObject*)&NOMStyle_Type) < 0) {
        return E_FAIL;
    }
    
    Py_INCREF(&NOMStyle_Type);
    if (PyModule_AddObject(m, "Style", (PyObject *)&NOMStyle_Type) < 0) {
        Py_DECREF(&NOMStyle_Type);
        return E_FAIL;
    }
    
    return S_OK;
}
