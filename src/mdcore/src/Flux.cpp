/*
 * Flux.cpp
 *
 *  Created on: Dec 21, 2020
 *      Author: andy
 */

#include <Flux.hpp>
#include "MxParticle.h"
#include "CSpeciesList.hpp"
#include "CStateVector.hpp"
#include <MxConvert.hpp>
#include <CConvert.hpp>
#include "flux_eval.hpp"
#include "space.h"
#include "space_cell.h"
#include "engine.h"



PyTypeObject MxFluxes_Type = {
    CObject_HEAD_INIT(NULL)
    "Fluxes"                              , // .tp_name
    sizeof(MxFluxes)                      , // .tp_basicsize
    sizeof(MxFlux)                        , // .tp_itemsize
    (destructor )0                        , // .tp_dealloc
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

PyTypeObject MxFlux_Type = {
    CVarObject_HEAD_INIT(NULL, 0)
    "Flux"                                , // .tp_name
    sizeof(MxFlux)                      , // .tp_basicsize
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

HRESULT _MxFlux_Init(PyObject* m) {
    if (PyType_Ready((PyTypeObject*)&MxFluxes_Type) < 0) {
        return E_FAIL;
    }
    
    Py_INCREF(&MxFluxes_Type);
    if (PyModule_AddObject(m, "Fluxes", (PyObject *)&MxFluxes_Type) < 0) {
        Py_DECREF(&MxFluxes_Type);
        return E_FAIL;
    }
    
    if (PyType_Ready((PyTypeObject*)&MxFlux_Type) < 0) {
        return E_FAIL;
    }
    
    Py_INCREF(&MxFlux_Type);
    if (PyModule_AddObject(m, "Flux", (PyObject *)&MxFlux_Type) < 0) {
        Py_DECREF(&MxFlux_Type);
        return E_FAIL;
    }
    
    return S_OK;
}

PyObject* MxFluxes_FluxEx(FluxKind kind, PyObject *_a, PyObject *_b,
                          const std::string& name, float k, float decay, float target) {
    
    MxParticleType *a = MxParticleType_Get(_a);
    MxParticleType *b = MxParticleType_Get(_b);
    
    if(!a || !b) {
        throw std::invalid_argument("Invalid particle types");
    }
    
    if(!a->species) {
        std::string msg = std::string("particle type ") + a->name + " does not have any defined species";
        throw std::invalid_argument(msg);
    }
    
    if(!b->species) {
        std::string msg = std::string("particle type ") + b->name + " does not have any defined species";
        throw std::invalid_argument(msg);
    }
    
    int index_a = a->species->index_of(name);
    int index_b = b->species->index_of(name);
    
    if(index_a < 0) {
        std::string msg = std::string("particle type ") +
        a->name + " does not have species " + name;
        throw std::invalid_argument(msg);
    }
    
    if(index_b < 0) {
        std::string msg = std::string("particle type ") +
        b->name + " does not have species " + name;
        throw std::invalid_argument(msg);
    }
    
    int fluxes_index1 = _Engine.max_type * a->id + b->id;
    int fluxes_index2 = _Engine.max_type * b->id + a->id;
    
    MxFluxes *fluxes = _Engine.fluxes[fluxes_index1];
    
    if(fluxes == NULL) {
        fluxes = MxFluxes_New(8);
    }
    
    fluxes = MxFluxes_AddFlux(kind, fluxes, a->id, b->id, index_a, index_b, k, decay, target);
    
    _Engine.fluxes[fluxes_index1] = fluxes;
    _Engine.fluxes[fluxes_index2] = fluxes;
    
    return Py_INCREF(fluxes), fluxes;
}

PyObject* MxFluxes_Fick(PyObject *self, PyObject *args, PyObject *kwargs) {
    try {
        PyObject *a =      mx::arg<PyObject*>("A", 0, args, kwargs);
        PyObject *b =      mx::arg<PyObject*>("B", 1, args, kwargs);
        std::string name = mx::arg<std::string>("name", 2, args, kwargs);
        float k =          mx::arg<float>("k", 3, args, kwargs);
        float decay =      mx::arg<float>("decay", 4, args, kwargs, 0.f);
        
        return MxFluxes_FluxEx(FLUX_FICK, a, b, name, k, decay, 0.f);
    }
    catch(const std::exception &e) {
        C_RETURN_EXP(e);
    }
}


PyObject* MxFluxes_Secrete(PyObject *self, PyObject *args, PyObject *kwargs) {
    try {
        PyObject *a =      mx::arg<PyObject*>("A", 0, args, kwargs);
        PyObject *b =      mx::arg<PyObject*>("B", 1, args, kwargs);
        std::string name = mx::arg<std::string>("name", 2, args, kwargs);
        float k =          mx::arg<float>("k", 3, args, kwargs);
        float target =     mx::arg<float>("target", 4, args, kwargs);
        float decay =      mx::arg<float>("decay", 5, args, kwargs, 0.f);
        
        return MxFluxes_FluxEx(FLUX_SECRETE, a, b, name, k, decay, target);
    }
    catch(const std::exception &e) {
        C_RETURN_EXP(e);
    }
}

PyObject* MxFluxes_Uptake(PyObject *self, PyObject *args, PyObject *kwargs) {
    try {
        PyObject *a =      mx::arg<PyObject*>("A", 0, args, kwargs);
        PyObject *b =      mx::arg<PyObject*>("B", 1, args, kwargs);
        std::string name = mx::arg<std::string>("name", 2, args, kwargs);
        float k =          mx::arg<float>("k", 3, args, kwargs);
        float target =     mx::arg<float>("target", 4, args, kwargs);
        float decay =      mx::arg<float>("decay", 5, args, kwargs, 0.f);
        
        return MxFluxes_FluxEx(FLUX_UPTAKE, a, b, name, k, decay, target);
    }
    catch(const std::exception &e) {
        C_RETURN_EXP(e);
    }
}

static void integrate_statevector(CStateVector *s) {
    for(int i = 0; i < s->size; ++i) {
        s->fvec[i] += _Engine.dt * s->q[i];
        s->q[i] = 0; // clear flux for next step
    }
}


HRESULT MxFluxes_Integrate(int cellId) {
    
    space_cell *c = &_Engine.s.cells[cellId];
    MxParticle *p;
    CStateVector *s;
    
    
    for(int i = 0; i < c->count; ++i) {
        p = &c->parts[i];
        s = p->state_vector;
        
        if(s) {
            integrate_statevector(s);
        }
    }
    
    return S_OK;
}

MxFluxes *MxFluxes_AddFlux(FluxKind kind, MxFluxes *fluxes,
                           int16_t typeId_a, int16_t typeId_b,
                           int32_t index_a, int32_t index_b,
                           float k, float decay, float target) {
    int i = 0;
    if(fluxes->size + 1 < fluxes->fluxes_size * MX_SIMD_SIZE) {
        i = fluxes->fluxes[0].size;
        fluxes->size += 1;
        fluxes->fluxes[0].size += 1;
    }
    
    MxFlux *flux = &fluxes->fluxes[0];
    
    flux->kinds[i] = kind;
    flux->type_ids[i].a = typeId_a;
    flux->type_ids[i].b = typeId_b;
    flux->indices_a[i] = index_a;
    flux->indices_b[i] = index_b;
    flux->coef[i] = k;
    flux->decay_coef[i] = decay;
    flux->target[i] = target;
    
    return fluxes;
}


MxFluxes *MxFluxes_New(int32_t init_size) {
    
    Log(LOG_TRACE);

    struct MxFluxes *obj = NULL;
    
    PyTypeObject *type = &MxFluxes_Type;
    
    int32_t blocks = std::ceil((double)init_size / MX_SIMD_SIZE);
    
    int total_size = type->tp_basicsize + blocks * type->tp_itemsize;

    /* allocate the potential */
    if ((obj = (MxFluxes * )CAligned_Malloc(total_size, 16 )) == NULL ) {
        return NULL;
    }
    
    ::memset(obj, 0, total_size);
    
    obj->size = 0;
    obj->fluxes_size = blocks;
    
    if (type->tp_flags & Py_TPFLAGS_HEAPTYPE)
        Py_INCREF(type);

    PyObject_INIT(obj, type);

    if (PyType_IS_GC(type)) {
        assert(0 && "should not get here");
        //  _PyObject_GC_TRACK(obj);
    }

    return obj;
}


