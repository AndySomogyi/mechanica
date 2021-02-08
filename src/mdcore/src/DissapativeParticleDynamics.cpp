/*
 * DissipativeParticleDynamics.cpp
 *
 *  Created on: Feb 7, 2021
 *      Author: andy
 */

#include <DissapativeParticleDynamics.hpp>
#include <MxConvert.hpp>


#define DPD_SELF(handle) DPDPotential *self = ((DPDPotential*)(handle))



static PyMethodDef dpd_methods[] = {
    { NULL, NULL, 0, NULL }
};


PyGetSetDef dpd_getsets[] = {
    {
        .name = "alpha",
        .get = [](PyObject *obj, void *p) -> PyObject* {
            DPD_SELF(obj);
            return mx::cast(self->alpha);
        },
        .set = [](PyObject *obj, PyObject *val, void *p) -> int {
            try {
                DPD_SELF(obj);
                self->alpha = mx::cast<float>(val);
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
        .name = "gamma",
        .get = [](PyObject *obj, void *p) -> PyObject* {
            DPD_SELF(obj);
            return mx::cast(self->gamma);
        },
        .set = [](PyObject *obj, PyObject *val, void *p) -> int {
            try {
                DPD_SELF(obj);
                self->gamma = mx::cast<float>(val);
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
        .name = "sigma",
        .get = [](PyObject *obj, void *p) -> PyObject* {
            DPD_SELF(obj);
            return mx::cast(self->sigma);
        },
        .set = [](PyObject *obj, PyObject *val, void *p) -> int {
            try {
                DPD_SELF(obj);
                self->sigma = mx::cast<float>(val);
                return 0;
            }
            catch (const std::exception &e) {
                return C_EXP(e);
            }
        },
        .doc = "test doc",
        .closure = NULL
    },
    {NULL}
};

PyTypeObject DPDPotential_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name =           "DPDPotential",
    .tp_basicsize =      sizeof(DPDPotential),
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
    .tp_methods =        dpd_methods,
    .tp_members =        0,
    .tp_getset =         dpd_getsets,
    .tp_base =           &MxPotential_Type,
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


PyObject * DPDPotential_New(float alpha, float gamma, float sigma, float cutoff) {
    DPDPotential *p = (DPDPotential*)potential_alloc(&DPDPotential_Type);
    
    p->alpha = alpha;
    p->gamma = gamma;
    p->sigma = sigma;
    p->b = cutoff;
    
    return p;
}


HRESULT _DPDPotential_Init(PyObject* m) {
        
    if (PyType_Ready((PyTypeObject*)&DPDPotential_Type) < 0) {
        return E_FAIL;
    }

    Py_INCREF(&DPDPotential_Type);
    if (PyModule_AddObject(m, "DPDPotential", (PyObject *)&DPDPotential_Type) < 0) {
        Py_DECREF(&DPDPotential_Type);
        return E_FAIL;
    }

    return S_OK;
}



