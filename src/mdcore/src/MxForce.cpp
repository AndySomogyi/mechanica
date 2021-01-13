/*
 * MxForce.cpp
 *
 *  Created on: May 21, 2020
 *      Author: andy
 */

#include <MxForce.h>
#include <engine.h>
#include <MxParticle.h>
#include <iostream>
#include <MxPy.h>
#include <random>

static PyObject *berenderson_create(float tau);
static PyObject *random_create(float std, float mean, float durration);

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
PyTypeObject MxForce_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name =           "Force",
    .tp_basicsize =      sizeof(MxForce),
    .tp_itemsize =       1,
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

static PyObject* py_berenderson_create(PyObject *m, PyObject *_args, PyObject *_kwds) {
    try {
        pybind11::detail::loader_life_support ls{};
        pybind11::args args = pybind11::reinterpret_borrow<pybind11::args>(_args);
        pybind11::kwargs kwargs = pybind11::reinterpret_borrow<pybind11::kwargs>(_kwds);

        float tau = arg<float>("tau", 0, args.ptr(), kwargs.ptr());

        return berenderson_create(tau);
    }
    catch (const pybind11::builtin_exception &e) {
        e.set_error();
        return NULL;
    }
}


static PyObject* py_random_create(PyObject *m, PyObject *_args, PyObject *_kwds) {
    try {
        pybind11::detail::loader_life_support ls{};
        pybind11::args args = pybind11::reinterpret_borrow<pybind11::args>(_args);
        pybind11::kwargs kwargs = pybind11::reinterpret_borrow<pybind11::kwargs>(_kwds);
        
        float std = arg<float>("std", 0, args.ptr(), kwargs.ptr());
        float mean = arg<float>("mean", 1, args.ptr(), kwargs.ptr());
        float durration = arg<float>("durration", 2, args.ptr(), kwargs.ptr(), 0.01);
        
        return random_create(std, mean, durration);
    }
    catch (const pybind11::builtin_exception &e) {
        e.set_error();
        return NULL;
    }
}




static PyMethodDef methods[] = {
    { "berenderson_tstat", (PyCFunction)py_berenderson_create, METH_VARARGS | METH_KEYWORDS, NULL},
    { "random", (PyCFunction)py_random_create, METH_VARARGS | METH_KEYWORDS, NULL},
    { NULL, NULL, 0, NULL }
};

static struct PyModuleDef forces_moduledef = {
        PyModuleDef_HEAD_INIT,
        "forces",   /* name of module */
        NULL, /* module documentation, may be NULL */
        -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
        methods
};

static PyObject *forces_module;

HRESULT _MxForces_init(PyObject *m)
{
     if (PyType_Ready((PyTypeObject*)&MxForce_Type) < 0) {
         std::cout << "could not initialize MxForce_Type " << std::endl;
         return E_FAIL;
     }

     forces_module = PyModule_Create(&forces_moduledef);

     Py_INCREF(&MxForce_Type);
     if (PyModule_AddObject(m, "Force", (PyObject *)&MxForce_Type) < 0) {
         Py_DECREF(&MxForce_Type);
         return E_FAIL;
     }

     if (PyModule_AddObject(m, "forces", (PyObject *)forces_module) < 0) {
         Py_DECREF(&MxForce_Type);
         Py_DECREF(&forces_module);
         return E_FAIL;
     }

     return S_OK;
}

struct Berendsen : MxForce {
    float itau;
};

struct Gaussian : MxForce {
    float std;
    float mean;
    unsigned durration_steps;
};

/**
 * Implements a force:
 *
 * f_b = p / tau * ((T_0 / T) - 1)
 */
static void berendsen_force(struct Berendsen* t, struct MxParticle *p, FPTYPE*f) {
    MxParticleType *type = (MxParticleType*)&engine::types[p->typeId];

    float scale = t->itau * ((type->target_energy / type->kinetic_energy) - 1.0);
    f[0] += scale * p->v[0];
    f[1] += scale * p->v[1];
    f[2] += scale * p->v[2];
}

static void gaussian_force(struct Gaussian* t, struct MxParticle *p, FPTYPE*f) {
    MxParticleType *type = (MxParticleType*)&engine::types[p->typeId];
    // values near the mean are the most likely
    // standard deviation affects the dispersion of generated values from the mean
    
    if((_Engine.integrator_flags & INTEGRATOR_UPDATE_PERSISTENTFORCE) &&
       (_Engine.time + p->id) % t->durration_steps == 0) {
        
        
        p->persistent_force = MxRandomVector(t->mean, t->std);
    }
    
    f[0] += p->persistent_force[0];
    f[1] += p->persistent_force[1];
    f[2] += p->persistent_force[2];
}

PyObject *berenderson_create(float tau) {
    Berendsen *obj = (Berendsen*)PyType_GenericAlloc(&MxForce_Type,
            sizeof(Berendsen) - sizeof(MxForce));

    obj->func = (MxForce_OneBodyPtr)berendsen_force;
    obj->itau = 1/tau;

    return (PyObject*)obj;
}

PyObject *random_create(float mean, float std, float durration) {
    Gaussian *obj = (Gaussian*)PyType_GenericAlloc(&MxForce_Type,
                                                     sizeof(Gaussian) - sizeof(MxForce));
    
    obj->func = (MxForce_OneBodyPtr)gaussian_force;
    obj->std = std;
    obj->mean = mean;
    obj->durration_steps = std::ceil(durration / _Engine.dt);
    
    return (PyObject*)obj;
}


