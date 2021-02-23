/*
 * MxForce.cpp
 *
 *  Created on: May 21, 2020
 *      Author: andy
 */

#include <MxForce.h>
#include <engine.h>
#include <MxParticle.h>
#include <MxConvert.hpp>
#include <iostream>
#include <MxPy.h>
#include <random>

static PyObject *berenderson_create(float tau);
static PyObject *random_create(float std, float mean, float durration);
static PyObject *friction_create(float coef, float mean, float std, float durration);


static int constantforce_init(MxConstantForce *self, PyObject *_args, PyObject *_kwds);

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
    .tp_new =            PyType_GenericNew,
    .tp_free =           [] (void* p) {
        Log(LOG_DEBUG) <<  "freeing force";
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



static PyGetSetDef constantforce_getset[] = {
    {
        .name = "value",
        .get = [](PyObject *obj, void *p) -> PyObject* {
            MxConstantForce *self = (MxConstantForce*)obj;
            if(self->userFunc) {
                Py_INCREF(self->userFunc);
                return self->userFunc;
            }
            else {
                return mx::cast(self->force);
            }
        },
        .set = [](PyObject *obj, PyObject *val, void *p) -> int {
            try {
                MxConstantForce *self = (MxConstantForce*)obj;
                self->setValue(val);
                return 0;
            }
            catch(const std::exception &e) {
                return C_EXP(e);
            }
        },
        .doc = "test doc",
        .closure = NULL
    },
    {
        .name = "period",
        .get = [](PyObject *obj, void *p) -> PyObject* {
            MxConstantForce *self = (MxConstantForce*)obj;
            return mx::cast(self->updateInterval);
        },
        .set = [](PyObject *obj, PyObject *val, void *p) -> int {
            try {
                MxConstantForce *self = (MxConstantForce*)obj;
                self->updateInterval = mx::cast<double>(val);
                return 0;
            }
            catch(const std::exception &e) {
                return C_EXP(e);
            }
        },
        .doc = "test doc",
        .closure = NULL
    },
    {NULL},
};


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
PyTypeObject MxConstantForce_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name =           "ConstantForce",
    .tp_basicsize =      sizeof(MxConstantForce),
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
    .tp_getset =         constantforce_getset,
    .tp_base =           &MxForce_Type,
    .tp_dict =           0,
    .tp_descr_get =      (descrgetfunc)0,
    .tp_descr_set =      0,
    .tp_dictoffset =     0,
    .tp_init =           (initproc)constantforce_init,
    .tp_alloc =          0,
    .tp_new =            0,
    .tp_free =           [] (void* p) {
        Log(LOG_DEBUG) << "freeing force";
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


static PyObject* py_berenderson_create(PyObject *m, PyObject *args, PyObject *kwargs) {
    try {

        float tau = mx::arg<float>("tau", 0, args, kwargs);

        return berenderson_create(tau);
    }
    catch (const std::exception &e) {
        C_RETURN_EXP(e);
    }
}


static PyObject* py_random_create(PyObject *m, PyObject *args, PyObject *kwargs) {
    try {
        float std = mx::arg<float>("std", 0, args, kwargs);
        float mean = mx::arg<float>("mean", 1, args, kwargs);
        float durration = mx::arg<float>("durration", 2, args, kwargs, 0.01);
        
        return random_create(std, mean, durration);
    }
    catch (const std::exception &e) {
        C_RETURN_EXP(e);
    }
}

static PyObject* py_friction_create(PyObject *m, PyObject *args, PyObject *kwargs) {
    try {
        float coef = mx::arg<float>("coef", 0, args, kwargs);
        float std = mx::arg<float>("std", 1, args, kwargs, 0);
        float mean = mx::arg<float>("mean", 2, args, kwargs, 0);
        float durration = mx::arg<float>("durration", 3, args, kwargs, 0.01);
        
        return friction_create(coef, std, mean, durration);
    }
    catch (const std::exception &e) {
        C_RETURN_EXP(e);
    }
}




static PyMethodDef methods[] = {
    { "berenderson_tstat", (PyCFunction)py_berenderson_create, METH_VARARGS | METH_KEYWORDS, NULL},
    { "random", (PyCFunction)py_random_create, METH_VARARGS | METH_KEYWORDS, NULL},
    { "friction", (PyCFunction)py_friction_create, METH_VARARGS | METH_KEYWORDS, NULL},
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
         return c_error(E_FAIL, "could not initialize MxForce_Type " );
     }
    
    if (PyType_Ready((PyTypeObject*)&MxConstantForce_Type) < 0) {
        return c_error(E_FAIL, "could not initialize MxConstantForce_Type " );
    }
    
     forces_module = PyModule_Create(&forces_moduledef);

     Py_INCREF(&MxForce_Type);
     if (PyModule_AddObject(forces_module, "Force", (PyObject *)&MxForce_Type) < 0) {
         Py_DECREF(&MxForce_Type);
         return E_FAIL;
     }
    
    Py_INCREF(&MxConstantForce_Type);
    if (PyModule_AddObject(forces_module, "ConstantForce", (PyObject *)&MxConstantForce_Type) < 0) {
        Py_DECREF(&MxConstantForce_Type);
        return E_FAIL;
    }

     if (PyModule_AddObject(m, "forces", (PyObject *)forces_module) < 0) {
         Py_DECREF(&MxForce_Type);
         Py_DECREF(&forces_module);
         Py_DECREF(&MxConstantForce_Type);
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

struct Friction : MxForce {
    float coef;
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

static void constant_force(struct MxConstantForce* cf, struct MxParticle *p, FPTYPE*f) {
    f[0] += cf->force[0];
    f[1] += cf->force[1];
    f[2] += cf->force[2];
}


/**
 * Implements a friction force:
 *
 * f_b = p / tau * ((T_0 / T) - 1)
 */
static void friction_force(struct Friction* t, struct MxParticle *p, FPTYPE*f) {
    MxParticleType *type = (MxParticleType*)&engine::types[p->typeId];
    
    if((_Engine.integrator_flags & INTEGRATOR_UPDATE_PERSISTENTFORCE) &&
       (_Engine.time + p->id) % t->durration_steps == 0) {
        
        
        p->persistent_force = MxRandomVector(t->mean, t->std);
    }
    
    float v2 = Magnum::Math::dot(p->velocity, p->velocity);
    float scale = -1. * t->coef * v2;
    
    f[0] += scale * p->v[0] + p->persistent_force[0];
    f[1] += scale * p->v[1] + p->persistent_force[1];
    f[2] += scale * p->v[2] + p->persistent_force[2];
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

PyObject *friction_create(float coef, float mean, float std, float durration) {
    Friction *obj = (Friction*)PyType_GenericAlloc(&MxForce_Type,
                                                   sizeof(Friction) - sizeof(MxForce));
    
    obj->func = (MxForce_OneBodyPtr)friction_force;
    obj->coef = coef;
    obj->std = std;
    obj->mean = mean;
    obj->durration_steps = std::ceil(durration / _Engine.dt);
    
    return (PyObject*)obj;
}

void MxConstantForce::onTime(double time)
{
    if(userFunc && time >= lastUpdate + updateInterval) {
        lastUpdate = time;
        setValue(userFunc);
    }
}

void MxConstantForce::setValue(PyObject *obj)
{
    if(PyCallable_Check(obj)) {
        
        PyObject *value = NULL;
                
        // check if method is valid
        if(PyFunction_Check(obj)) {
            
            PyObject *args = PyTuple_New(0);
            
            value = PyObject_CallObject(obj, args);
            
            Py_DECREF(args);
            
            if(value == NULL) {
                Py_DecRef(userFunc);
                userFunc = NULL;
                throw std::runtime_error("Error calling user function" + carbon::str(obj));
            }
        }
        else if(PyMethod_Check(obj)) {
            
            PyObject* function = PyMethod_Function(obj);
            
            PyObject* self = PyMethod_Self(obj);
            
            PyObject *args = PyTuple_Pack(1, self);
            
            value = PyObject_CallObject(function, args);
            
            Py_DECREF(args);
            
            if(value == NULL) {
                Py_DecRef(userFunc);
                userFunc = NULL;
                throw std::runtime_error("Error calling user function" + carbon::str(obj));
            }
        }

        // we got value back from a python func, need to decref it,
        // cast will throw exception if fails
        try {
            force = mx::cast<Magnum::Vector3>(value);
            Py_DecRef(value);
        }
        catch(const std::exception &e) {
            Py_DecRef(value);
            Py_DecRef(userFunc);
            userFunc = NULL;
            throw e;
        }
        
        if(userFunc != obj) {
            if(userFunc) {
                Py_DECREF(userFunc);
            }
            userFunc = obj;
            Py_INCREF(userFunc);
        }
    }
    else {
        force = mx::cast<Magnum::Vector3>(obj);
    }
}

int MxConstantForce_Check(PyObject *o) {
    return PyObject_IsInstance(o, (PyObject*)&MxConstantForce_Type);
}

/**
 * initialize a constant force object.
 *
 * value: Vector3, or python callable.
 * period: update period of object, 
 */
int constantforce_init(MxConstantForce *self, PyObject *args, PyObject *kwds) {
    try {
        self->setValue(mx::arg<PyObject*>("value", 0, args, kwds));
        self->updateInterval = mx::arg<float>("period", 1, args, kwds, std::numeric_limits<float>::max());
        self->func = (MxForce_OneBodyPtr)constant_force;
        return 0;
    }
    catch (const std::exception &e) {
        return C_EXP(e);
    }
}
