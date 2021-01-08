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

struct MxFlux : PyObject {
    
};



PyTypeObject MxFluxes_Type = {
    CObject_HEAD_INIT(NULL)
    "Fluxes"                              , // .tp_name
    sizeof(MxFluxes)                      , // .tp_basicsize
                                            // .tp_itemsize
    2 * sizeof(int32_t) + 2 * sizeof(float),
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

PyObject* MxFluxes_FluxEx(PyObject *_a, PyObject *_b, PyObject *sname, PyObject *_k, PyObject *_decay) {
    
    MxParticleType *a = MxParticleType_Get(_a);
    MxParticleType *b = MxParticleType_Get(_b);
    
    if(!a || !b) {
        PyErr_SetString(PyExc_TypeError, "Invalid particle types");
        return NULL;
    }
    
    if(!carbon::check<float>(_k)) {
        PyErr_SetString(PyExc_TypeError, "flux constant k must be a number");
        return NULL;
    }
    
    float k = carbon::cast<float>(_k);
    
    float decay = _decay ? carbon::cast<float>(_decay) : 0;
    
    if(!a->species) {
        std::string msg = std::string("particle type ") + a->name + " does not have any defined species";
        PyErr_SetString(PyExc_ValueError, msg.c_str());
        return NULL;
    }
    
    if(!b->species) {
        std::string msg = std::string("particle type ") + b->name + " does not have any defined species";
        PyErr_SetString(PyExc_ValueError, msg.c_str());
        return NULL;
    }

    int index_a = a->species->index_of(sname);
    int index_b = b->species->index_of(sname);
    
    if(index_a < 0) {
        std::string msg = std::string("particle type ") +
        a->name + " does not have species " + carbon::cast<std::string>(sname);
        PyErr_SetString(PyExc_ValueError, msg.c_str());
        return NULL;
    }
    
    if(index_b < 0) {
         std::string msg = std::string("particle type ") +
         b->name + " does not have species " + carbon::cast<std::string>(sname);
         PyErr_SetString(PyExc_ValueError, msg.c_str());
         return NULL;
     }
    
    int fluxes_index1 = _Engine.max_type * a->id + b->id;
    int fluxes_index2 = _Engine.max_type * b->id + a->id;
    
    MxFluxes *fluxes1 = _Engine.fluxes[fluxes_index1];
    MxFluxes *fluxes2 = _Engine.fluxes[fluxes_index2];
    
    if(fluxes1 == NULL) {
        fluxes1 = MxFluxes_New(8);
    }
    
    if(fluxes2 == NULL) {
        fluxes2 = MxFluxes_New(8);
    }
    
    fluxes1 = MxFluxes_AddFlux(fluxes1,  index_a,  index_b,  k, decay);
    fluxes2 = MxFluxes_AddFlux(fluxes2,  index_b,  index_a,  k, decay);
    
    // TODO: probably need to flip order of indexa, indexb...
    _Engine.fluxes[fluxes_index1] = fluxes1;
    _Engine.fluxes[fluxes_index2] = fluxes2;
    
    
    return Py_INCREF(fluxes1), fluxes1;
}

PyObject* MxFluxes_FluxPy(PyObject *args, PyObject *kwargs) {
    PyObject *decay = NULL;
    
    if(PyTuple_Size(args) < 4) {
        PyErr_SetString(PyExc_TypeError, "invalid number of args, flux(A, B, species_name, k)");
    }
    
    if(PyTuple_Size(args) >= 5) {
        decay = PyTuple_GetItem(args, 4);
    }
    
    return MxFluxes_FluxEx(PyTuple_GetItem(args, 0),
                           PyTuple_GetItem(args, 1),
                           PyTuple_GetItem(args, 2),
                           PyTuple_GetItem(args, 3),
                           decay);
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

MxFluxes *MxFluxes_AddFlux(MxFluxes *fluxes, int32_t index_a, int32_t index_b, float k, float decay) {
    int i = 0;
    if(fluxes->size + 1 < fluxes->alloc_size) {
        i = fluxes->size;
        fluxes->size += 1;
    }
    
    fluxes->indices_a[i] = index_a;
    fluxes->indices_b[i] = index_b;
    fluxes->coef[i] = k;
    fluxes->decay_coef[i] = decay;
    
    return fluxes;
}


MxFluxes *MxFluxes_New(int32_t init_size) {
    
    std::cout << MX_FUNCTION << std::endl;

    struct MxFluxes *obj = NULL;
    
    PyTypeObject *type = &MxFluxes_Type;
    
    int total_size = type->tp_basicsize + init_size * type->tp_itemsize;

    /* allocate the potential */
    if ((obj = (MxFluxes * )CAligned_Malloc(total_size, 16 )) == NULL ) {
        return NULL;
    }
    
    ::memset(obj, NULL, total_size);
    
    obj->alloc_size = init_size;
    
    /*
    MxFluxes *test = &obj[0];
    MxFluxes *test2 = &obj[1];
    float *pf = (float*)obj;
    int32_t *pi = (int32_t*)obj;
     */

    obj->indices_a  = (int32_t*)((std::byte*)obj + type->tp_basicsize);
    obj->indices_b  = (int32_t*)((std::byte*)obj + type->tp_basicsize + init_size * sizeof(int32_t));
    obj->coef       = (float*)  ((std::byte*)obj + type->tp_basicsize + 2 * init_size * sizeof(int32_t));
    obj->decay_coef = (float*)  ((std::byte*)obj + type->tp_basicsize + 2 * init_size * sizeof(int32_t) + init_size * sizeof(float));
    
    /*
    std::cout << "diff: " << (std::byte*)obj->indices_a - (std::byte*)obj << std::endl;
    std::cout << "diff: " << (std::byte*)test2 - (std::byte*)obj << std::endl;
    for(int i = 0; i < init_size; ++i) {
        obj->indices_a[i] = i;
        obj->indices_b[i] = i;
        obj->coef[i] = i;
    }
    
    for(int i = 0; i < total_size / 4; ++i) {
        std::cout << "i: " << i << ", fval: " << pf[i] << ", ival: " << pi[i] << std::endl;
    }
    */
    

    if (type->tp_flags & Py_TPFLAGS_HEAPTYPE)
        Py_INCREF(type);

    PyObject_INIT(obj, type);

    if (PyType_IS_GC(type)) {
        assert(0 && "should not get here");
        //  _PyObject_GC_TRACK(obj);
    }

    return obj;
}


