/*
 * MxUniverse.cpp
 *
 *  Created on: Sep 7, 2019
 *      Author: andy
 */
#ifdef _WIN32
#define _USE_MATH_DEFINES
#endif

#include <MxUniverse.h>
#include <MxForce.h>
#include <MxPy.h>
#include <MxSimulator.h>
#include <MxConvert.hpp>
#include <MxUtil.h>
#include <metrics.h>
#include <cmath>
#include <iostream>
#include <limits>
#include <MxThreadPool.hpp>
#include <MxCuboid.hpp>
#include <MxBind.hpp>
#include <MxPy.h>
#include <CStateVector.hpp>
#include "Magnum/Math/Matrix4.h"


#define PY_CHECK(hr) {if(!SUCCEEDED(hr)) { throw py::error_already_set();}}

static void print_performance_counters();

using Magnum::Vector3;

MxUniverse Universe = {
    .isRunning = false
};



static HRESULT universe_bind_force(MxForce *f, PyObject *a);

static PyObject *universe_virial(PyObject *self, PyObject *_args, PyObject *_kwargs);

static Magnum::Vector3 universe_center();

// the single static engine instance per process

// complete and total hack to get the global engine to show up here
// instead of the mdcore static lib.
// TODO: fix this crap.
engine _Engine = {
        .flags = 0
};

// default to paused universe
static uint32_t universe_flags = 0;


CAPI_FUNC(struct engine*) engine_get()
{
    return &_Engine;
}


// TODO: fix error handling values
#define UNIVERSE_CHECKERROR() { \
    if (_Engine.flags == 0 ) { \
        std::string err = "Error in "; \
        err += MX_FUNCTION; \
        err += ", Universe not initialized"; \
        return mx_error(E_FAIL, err.c_str()); \
    } \
    }

#define UNIVERSE_TRY() \
    try {\
        if(_Engine.flags == 0) { \
            std::string err = MX_FUNCTION; \
            err += "universe not initialized"; \
            throw std::domain_error(err.c_str()); \
        }

#define UNIVERSE_CHECK(hr) \
    if(SUCCEEDED(hr)) { Py_RETURN_NONE; } \
    else {return NULL;}

#define UNIVERSE_FINALLY(retval) \
    } \
    catch(const std::exception &e) { \
        C_EXP(e); return retval; \
    }

MxUniverseConfig::MxUniverseConfig() :
    origin {0, 0, 0},
    dim {10, 10, 10},
    spaceGridSize {4, 4, 4},
    cutoff{1},
    flags{0},
    maxTypes{64},
    dt{0.01},
    temp{1},
    nParticles{100},
    threads{mx::ThreadPool::hardwareThreadSize()},
    integrator{EngineIntegrator::FORWARD_EULER},
    boundaryConditionsPtr{NULL},
    max_distance{-1},
    timers_mask {0},
    timer_output_period {-1}
{
}


static PyObject *universe_bind(PyObject *self, PyObject *args, PyObject *kwargs) {
    UNIVERSE_TRY();
    PyObject *result = NULL;
    UNIVERSE_CHECK(MxUniverse_Bind(args, kwargs, &result));
    Py_RETURN_NONE;
    UNIVERSE_FINALLY(NULL);
}

static PyObject *universe_step(PyObject *self, PyObject *args, PyObject *kwargs) {
    UNIVERSE_TRY();
    double until = mx::arg<double>("until", 0, args, kwargs, 0);
    double dt = mx::arg<double>("dt", 1, args, kwargs, 0);
    UNIVERSE_CHECK(MxUniverse_Step(until, dt));
    Py_RETURN_NONE;
    UNIVERSE_FINALLY(NULL);
}

static PyObject *universe_stop(PyObject *self, PyObject *args, PyObject *kwargs) {
    UNIVERSE_TRY();
    UNIVERSE_CHECK(MxUniverse_SetFlag(MxUniverse_Flags::MX_RUNNING, false));
    Py_RETURN_NONE;
    UNIVERSE_FINALLY(NULL);
}

static PyObject *universe_start(PyObject *self, PyObject *args, PyObject *kwargs) {
    UNIVERSE_TRY();
    UNIVERSE_CHECK(MxUniverse_SetFlag(MxUniverse_Flags::MX_RUNNING, true));
    Py_RETURN_NONE;
    UNIVERSE_FINALLY(NULL);
}

static PyObject *universe_bind_pairwise(PyObject *self, PyObject *args, PyObject *kwargs) {
    UNIVERSE_TRY();
    return MxPyUniverse_BindPairwise(args, kwargs);
    UNIVERSE_FINALLY(NULL);
}

static PyObject *universe_reset(PyObject *self, PyObject *args, PyObject *kwargs) {
    UNIVERSE_TRY();
    engine_reset(&_Engine);
    Py_RETURN_NONE;
    UNIVERSE_FINALLY(NULL);
}

static PyObject *universe_particles(PyObject *self) {
    UNIVERSE_TRY();
    return (PyObject*)MxParticleList_All();
    UNIVERSE_FINALLY(NULL);
}

static PyMethodDef universe_methods[] = {
    { "bind", (PyCFunction)universe_bind, METH_STATIC| METH_VARARGS | METH_KEYWORDS, NULL },
    { "bind_pairwise", (PyCFunction)universe_bind_pairwise, METH_STATIC| METH_VARARGS | METH_KEYWORDS, NULL },
    { "virial", (PyCFunction)universe_virial, METH_STATIC| METH_VARARGS | METH_KEYWORDS, NULL },
    { "step", (PyCFunction)universe_step, METH_STATIC| METH_VARARGS | METH_KEYWORDS, NULL },
    { "stop", (PyCFunction)universe_stop, METH_STATIC| METH_VARARGS | METH_KEYWORDS, NULL },
    { "start", (PyCFunction)universe_start, METH_STATIC| METH_VARARGS | METH_KEYWORDS, NULL },
    { "reset", (PyCFunction)universe_reset, METH_STATIC| METH_VARARGS | METH_KEYWORDS, NULL },
    { "particles", (PyCFunction)universe_particles, METH_STATIC | METH_NOARGS, NULL },
    { "reset_species", (PyCFunction)MxUniverse_ResetSpecies, METH_STATIC| METH_VARARGS | METH_KEYWORDS, NULL },
    { NULL, NULL, 0, NULL }
};


PyGetSetDef universe_getsets[] = {
    {
        .name = "dim",
        .get = [](PyObject *obj, void *p) -> PyObject* {
            UNIVERSE_TRY();
            return mx::cast(MxUniverse::dim());
            UNIVERSE_FINALLY(0);
        },
        .set = [](PyObject *obj, PyObject *val, void *p) -> int {
            PyErr_SetString(PyExc_PermissionError, "read only");
            return -1;
        },
        .doc = "test doc",
        .closure = NULL
    },
    {
        .name = "temperature",
        .get = [](PyObject *obj, void *p) -> PyObject* {
            UNIVERSE_TRY();
            return mx::cast(engine_temperature(&_Engine));
            UNIVERSE_FINALLY(0);
        },
        .set = [](PyObject *obj, PyObject *val, void *p) -> int {
            PyErr_SetString(PyExc_PermissionError, "read only");
            return -1;
        },
        .doc = "test doc",
        .closure = NULL
    },
    {
        .name = "time",
        .get = [](PyObject *obj, void *p) -> PyObject* {
            UNIVERSE_TRY();
            return mx::cast(_Engine.time * _Engine.dt);
            UNIVERSE_FINALLY(0);
        },
        .set = [](PyObject *obj, PyObject *val, void *p) -> int {
            PyErr_SetString(PyExc_PermissionError, "read only");
            return -1;
        },
        .doc = "test doc",
        .closure = NULL
    },
    {
        .name = "dt",
        .get = [](PyObject *obj, void *p) -> PyObject* {
            UNIVERSE_TRY();
            return mx::cast(_Engine.dt);
            UNIVERSE_FINALLY(0);
        },
        .set = [](PyObject *obj, PyObject *val, void *p) -> int {
            PyErr_SetString(PyExc_PermissionError, "read only");
            return -1;
        },
        .doc = "test doc",
        .closure = NULL
    },
    {
        .name = "on_time",
        .get = [](PyObject *obj, void *p) -> PyObject* {
            UNIVERSE_TRY();
            Py_INCREF(_Engine.on_time); return (PyObject*)_Engine.on_time;
            UNIVERSE_FINALLY(0);
        },
        .set = [](PyObject *obj, PyObject *val, void *p) -> int {
            PyErr_SetString(PyExc_PermissionError, "read only");
            return -1;
        },
        .doc = "test doc",
        .closure = NULL
    },
    {
        .name = "center",
        .get = [](PyObject *obj, void *p) -> PyObject* {
            UNIVERSE_TRY();
            Magnum::Vector3 center = universe_center();
            PyObject *result = mx::cast(center);
            return result;
            UNIVERSE_FINALLY(0);
        },
        .set = [](PyObject *obj, PyObject *val, void *p) -> int {
            PyErr_SetString(PyExc_PermissionError, "read only");
            return -1;
        },
        .doc = "test doc",
        .closure = NULL
    },
    {
        .name = "boundary_conditions",
        .get = [](PyObject *obj, void *p) -> PyObject* {
            UNIVERSE_TRY();
            Py_IncRef((PyObject*)&_Engine.boundary_conditions);
            return (PyObject*)&_Engine.boundary_conditions;
            UNIVERSE_FINALLY(0);
        },
        .set = [](PyObject *obj, PyObject *val, void *p) -> int {
            PyErr_SetString(PyExc_PermissionError, "read only");
            return -1;
        },
        .doc = "test doc",
        .closure = NULL
    },
    {
        .name = "kinetic_energy",
        .get = [](PyObject *obj, void *p) -> PyObject* {
            UNIVERSE_TRY();
            return mx::cast(engine_kinetic_energy(&_Engine));
            UNIVERSE_FINALLY(0);
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



PyTypeObject MxUniverse_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name =           "MxUniverse",
    .tp_basicsize =      sizeof(PyObject),
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
    .tp_doc = "Custom objects",
    .tp_traverse =       0,
    .tp_clear =          0,
    .tp_richcompare =    0,
    .tp_weaklistoffset = 0,
    .tp_iter =           0,
    .tp_iternext =       0,
    .tp_methods =        universe_methods,
    .tp_members =        0,
    .tp_getset =         universe_getsets,
    .tp_base =           0,
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

Magnum::Vector3 MxUniverse::origin()
{
    return Vector3{(float)_Engine.s.origin[0], (float)_Engine.s.origin[1], (float)_Engine.s.origin[2]};
}

Magnum::Vector3 MxUniverse::dim()
{
    return Vector3{(float)_Engine.s.dim[0], (float)_Engine.s.dim[1], (float)_Engine.s.dim[2]};
}


HRESULT _MxUniverse_init(PyObject* m)
{
    if (PyType_Ready((PyTypeObject*)&MxUniverse_Type) < 0) {
        return E_FAIL;
    }
    
    PyObject* universe = PyObject_New(PyObject, &MxUniverse_Type);

    if(!universe) {
        return c_error(E_FAIL, "could not get universe from main module");
    }

    PyModule_AddObject(m, "Universe", universe);
    
    PyModule_AddObject(m, "universe", universe);

    return S_OK;
}


HRESULT MxUniverse_Bind(PyObject *args, PyObject *kwargs, PyObject **out)
{
    *out = NULL;

    if(kwargs && PyDict_Size(kwargs) > 0) {
        if((PyDict_Size(kwargs) > 1) ||
            PyUnicode_CompareWithASCIIString(PyList_GET_ITEM(PyDict_Keys(kwargs), 0), "bound") != 0) {
            return c_error(E_INVALIDARG, "Error, kwargs to Universe.bind contains invalid items");
        }
    }

    if(args && PyTuple_Size(args) == 4) {
        return MxUniverse_BindThing3(PyTuple_GetItem(args, 0),
                                     PyTuple_GetItem(args, 1),
                                     PyTuple_GetItem(args, 2),
                                     PyTuple_GetItem(args, 3));
    }

    PyObject *cutoff = NULL;
    MxParticleList *pl = NULL;
    if(args && PyTuple_Size(args) == 3 &&
       kwargs &&
       (cutoff = PyDict_GetItemString(kwargs, "cutoff")) &&
       (pl = MxParticleList_FromPyObject(PyTuple_GetItem(args, 1)))) {
        PyObject *pot = PyTuple_GetItem(args, 0);
        if(MxPotential_Check(pot) && PyNumber_Check(cutoff)) {
            PyObject *result = MxBond_PairwiseNew((MxPotential*)pot, pl,
                PyFloat_AsDouble(cutoff), NULL, args, kwargs);
            *out = result;
            Py_DECREF(pl);
            return S_OK;
        }
        Py_DecRef(pl);
    }

    PyObject *bound = NULL;
    if(args && PyTuple_Size(args) == 3 &&
       kwargs && (bound = PyDict_GetItemString(kwargs, "bound"))) {
        return MxUniverse_BindThing3(PyTuple_GetItem(args, 0),
                                     PyTuple_GetItem(args, 1),
                                     PyTuple_GetItem(args, 2),
                                     bound);
    }

    if(args && PyTuple_Size(args) == 3) {
        return MxUniverse_BindThing2(PyTuple_GetItem(args, 0),
                                     PyTuple_GetItem(args, 1),
                                     PyTuple_GetItem(args, 2));
    }

    if(args && PyTuple_Size(args) == 2) {
        return MxUniverse_BindThing1(PyTuple_GetItem(args, 0),
                                     PyTuple_GetItem(args, 1));
    }



    return mx_error(E_FAIL, "bind only implemented for 2 or 3 arguments: bind(thing, a, b)");
}


CAPI_FUNC(HRESULT) MxUniverse_BindThing3(PyObject *thing, PyObject *a,
                                         PyObject *b, PyObject *c)
{
    if(PyObject_IsInstance(thing, (PyObject*)&MxPotential_Type) && PyBool_Check(c)) {
        return universe_bind_potential((MxPotential*)thing, a, b, c == Py_True);
    }
    return mx_error(E_NOTIMPL, "binding currently implmented for potentials to things");
}

CAPI_FUNC(HRESULT) MxUniverse_BindThing2(PyObject *thing, PyObject *a,
        PyObject *b)
{
    if(PyObject_IsInstance(thing, (PyObject*)&MxPotential_Type)) {
        return universe_bind_potential((MxPotential*)thing, a, b);
    }
    return mx_error(E_NOTIMPL, "binding currently implmented for potentials to things");
}

CAPI_FUNC(HRESULT) MxUniverse_BindThing1(PyObject *thing, PyObject *a) {
    if(PyObject_IsInstance(thing, (PyObject*)&MxForce_Type)) {
        return universe_bind_force((MxForce*)thing, a);
    }
    return mx_error(E_NOTIMPL, "binding currently implmented for potentials to things");
}




static HRESULT universe_bind_force(MxForce *f, PyObject *a) {
    MxParticleType *a_type = MxParticleType_Get(a);
    if(a_type) {
        if(engine_addforce1(&_Engine, f, a_type->id) != engine_err_ok) {
            std::string msg = "failed to add force to engine: error";
            msg += std::to_string(engine_err);
            msg += ", ";
            msg += engine_err_msg[-engine_err];
            return mx_error(E_FAIL, msg.c_str());
        }
        return S_OK;
    }
    return mx_error(E_FAIL, "can only add force to particle types");
}


CAPI_FUNC(HRESULT) MxUniverse_Step(double until, double dt) {

    if(engine_err != 0) {
        return E_FAIL;
    }

    if ( engine_step( &_Engine ) != 0 ) {
        printf("main: engine_step failed with engine_err=%i.\n",engine_err);
        errs_dump(stdout);
        // TODO: correct error reporting
        return E_FAIL;
    }

    MxSimulator_Redraw();

    if(_Engine.timer_output_period > 0 && _Engine.time % _Engine.timer_output_period == 0 ) {
        print_performance_counters();
    }

    return S_OK;
}

// TODO: does it make sense to return an hresult???
int MxUniverse_Flag(MxUniverse_Flags flag)
{
    UNIVERSE_CHECKERROR();
    return universe_flags & flag;
}

CAPI_FUNC(HRESULT) MxUniverse_SetFlag(MxUniverse_Flags flag, int value)
{
    UNIVERSE_CHECKERROR();

    if(value) {
        universe_flags |= flag;
    }
    else {
        universe_flags &= ~(flag);
    }

    return MxSimulator_Redraw();
}


double ms(ticks tks)
{
    return (double)tks / (_Engine.time * CLOCKS_PER_SEC);
}


void print_performance_counters() {
    std::cout << "performance_timers : { " << std::endl;
    std::cout << "\t name:" << Universe.name << std::endl;
    std::cout << "\t fps: " << _Engine.time / _Engine.wall_time << std::endl;
    std::cout << "\t kinetic energy: " << engine_kinetic_energy(&_Engine) << std::endl;
    std::cout << "\t step: " << ms(_Engine.timers[engine_timer_step]) << std::endl;
    std::cout << "\t nonbond: " << ms(_Engine.timers[engine_timer_nonbond]) << std::endl;
    std::cout << "\t bonded: " << ms(_Engine.timers[engine_timer_bonded]) << std::endl;
    std::cout << "\t advance: " << ms(_Engine.timers[engine_timer_advance]) << std::endl;
    std::cout << "\t rendering: " << ms(_Engine.timers[engine_timer_render]) << std::endl;
    std::cout << "\t total: " << ms(_Engine.timers[engine_timer_render] + _Engine.timers[engine_timer_step]) << std::endl;
    std::cout << "\t time_steps: " << _Engine.time  << std::endl;
    std::cout << "}" << std::endl;
}


PyObject *universe_virial(PyObject *self, PyObject *_args, PyObject *_kwargs) {
    try {
        PyObject *_origin = mx::py_arg("origin", 0, _args, _kwargs);
        PyObject *_radius = mx::py_arg("radius", 1, _args, _kwargs);
        PyObject *_types = mx::py_arg("types", 2, _args, _kwargs);

        Magnum::Vector3 origin;
        float radius = 0;
        std::set<short int> typeIds;

        if(_origin) {
            origin = mx::cast<Magnum::Vector3>(_origin);
        }
        else {
            origin = universe_center();
        }

        if(_radius) {
            radius = mx::cast<float>(_radius);
        }
        else {
            // complete simulation domain
            radius = 2 * origin.max();
        }

        if(_types) {
            if(PyList_Check(_types)) {
                for(int i = 0; i < PyList_GET_SIZE(_types); ++i) {
                    PyObject *item = PyList_GET_ITEM(_types, i);
                    MxParticleType *type = MxParticleType_Get(item);
                    if(type) {
                        typeIds.insert(type->id);
                    }
                    else {
                        std::string msg = "error, types must be a list of Particle types, types[";
                        msg += std::to_string(i);
                        msg += "] is a ";
                        msg += item->ob_type->tp_name;
                        throw std::logic_error(msg.c_str());
                    }
                }
            }
            else {
                throw std::logic_error("types must be a list of Particle types");
            }
        }
        else {
            for(int i = 0; i < _Engine.nr_types; ++i) {
                typeIds.insert(i);
            }
        }
        Magnum::Matrix3 m;
        HRESULT result = MxCalculateVirial(origin.data(),
                                             radius,
                                             typeIds,
                                             m.data());
        if(SUCCEEDED(result)) {
            return mx::cast(m);
        }
    }
    catch(const std::exception &e) {
        C_RETURN_EXP(e);
    }
    return NULL;
}

Magnum::Vector3 universe_center() {
    Magnum::Vector3 center = {
        (float)_Engine.s.dim[0],
        (float)_Engine.s.dim[1],
        (float)_Engine.s.dim[2]
    };
    center = center / 2;
    return center;
}


/**
 * bind_pairwise(pot, parts, cutoff, *args, **kwargs)
 */
PyObject *MxPyUniverse_BindPairwise(PyObject *args, PyObject *kwds) {
    static const char* names[] = {
        "potential", "particles", "cutoff", "pairs"
    };

    try {
        PyObject *ppot = mx::py_arg(names[0], 0, args, kwds);
        PyObject *pparts = mx::py_arg(names[1], 1, args, kwds);
        PyObject *pcutoff = mx::py_arg(names[2], 2, args, kwds);
        PyObject *pairs = mx::py_arg(names[3], 3, args, kwds);

        MxPotential *pot;
        MxParticleList *parts;
        float cutoff;

        if(MxPotential_Check(ppot)) {
            pot = (MxPotential*)ppot;
        }
        else {
            c_error(E_FAIL, "argument 0 is not a potential");
            return NULL;
        }

        if((parts = MxParticleList_FromPyObject(pparts)) == NULL) {
            c_error(E_FAIL, "argument 1 is not a particle list");
            return NULL;
        }

        if(PyNumber_Check(pcutoff))  {
            cutoff = PyFloat_AsDouble(pcutoff);
        }
        else {
            c_error(E_FAIL, "argument 2 is not a number");
            return NULL;
        }

        PyObject *bond_args = NULL;

        // first 3 positional args, if given have to be pot, parts, and cutoff.
        // if we have more than 3 position args, pass those as additional args to
        // bonds ctor.

        if(PyTuple_Size(args) > 3) {
            bond_args = PyTuple_GetSlice(args, 3, PyTuple_Size(args));
        }
        else {
            bond_args = PyTuple_New(0);
        }

        if(kwds) {
            for(int i = 0; i < 4; ++i) {
                PyObject *key = PyUnicode_FromString(names[i]);
                if(PyDict_Contains(kwds, key)) {
                    PyDict_DelItem(kwds, key);
                }
                Py_DECREF(key);
            }
        }

        PyObject *result = MxBond_PairwiseNew(pot, parts, cutoff, pairs, bond_args, kwds);

        Py_DECREF(bond_args);

        return result;
    }
    catch (const std::exception &e) {
        C_RETURN_EXP(e);
    }
}

static bool contains_bond(PyListObject *bonds, int nbonds, int a, int b) {
    for(int i = 0; i < nbonds; ++i) {
        assert(i < PyList_GET_SIZE(bonds));
        MxBondHandle *h = (MxBondHandle*)PyList_GET_ITEM(bonds, i);
        MxBond *bond = &_Engine.bonds[h->id];
        if((bond->i == a && bond->j == b) || (bond->i == b && bond->j == a)) {
            return true;
        }
    }
    return false;
}

static int insert_bond(PyListObject *bonds, int nbonds, int a, int b,
                       MxPotential *pot, MxParticleList *parts) {
    int p1 = parts->parts[a];
    int p2 = parts->parts[b];
    if(!contains_bond(bonds, nbonds, p1, p2)) {
        MxBondHandle *bond = MxBondHandle_New(0, p1, p2,
                                              std::numeric_limits<double>::max(),
                                              std::numeric_limits<double>::max(),
                                              pot);
        assert(nbonds < PyList_GET_SIZE(bonds));
        PyList_SET_ITEM(bonds, nbonds, bond);
        return 1;
    }
    return 0;
}


PyObject* MxUniverse_BindSphere(PyObject *args, PyObject *kwds) {
    static const float Pi = M_PI;

    try {
        //potential
        //*     number of subdivisions
        //*     tuple of starting / stopping theta (polar angle)
        //*     center of sphere
        //*     radius of sphere
        PyObject *ppot = mx::py_arg("potential", 0, args, kwds);
        PyObject *pn = mx::py_arg("n", 1, args, kwds);
        PyObject *pcenter = mx::py_arg("center", 2, args, kwds);
        PyObject *pradius = mx::py_arg("radius", 3, args, kwds);
        PyObject *pphi = mx::py_arg("phi", 4, args, kwds);
        PyObject *type = mx::py_arg("type", 5, args, kwds);

        MxPotential *pot;
        int n;
        float phi0 = 0;
        float phi1 = Pi;

        if(ppot == NULL || !MxPotential_Check(ppot)) {
            throw std::logic_error("no potential given");
        }

        if(pn == NULL || !PyNumber_Check(pn)) {
            throw std::logic_error("no n number of subdivisions, or n not a number");
        }

        if(pphi) {
            if(!PyTuple_Check(pphi) || PyTuple_Size(pphi) != 2) {
                throw std::logic_error("phi must be a tuple (phi_0, phi_1)");
            }
            phi0 = mx::cast<float>(PyTuple_GetItem(pphi, 0));
            phi1 = mx::cast<float>(PyTuple_GetItem(pphi, 1));

            if(phi0 < 0 || phi0 > Pi) throw std::logic_error("phi_0 must be between 0 and pi");
            if(phi1 < 0 || phi1 > Pi) throw std::logic_error("phi_1 must be between 0 and pi");
            if(phi1 < phi0) throw std::logic_error("phi_1 must be greater than phi_0");
        }

        Magnum::Vector3 center =  pcenter ? mx::cast<Magnum::Vector3>(pcenter) : universe_center();

        float radius = pradius ? mx::cast<float>(pradius) : 1;

        pot = (MxPotential*)ppot;
        n = PyLong_AsLong(pn);

        std::vector<Magnum::Vector3> vertices;
        std::vector<int32_t> indices;

        Magnum::Matrix4 s = Magnum::Matrix4::scaling(Magnum::Vector3{radius, radius, radius});
        Magnum::Matrix4 t = Magnum::Matrix4::translation(center);
        Magnum::Matrix4 m = t * s;

        Mx_Icosphere(n, phi0, phi1, vertices, indices);

        Magnum::Vector3 velocity;

        MxParticleList *parts = MxParticleList_New(vertices.size());
        parts->nr_parts = vertices.size();

        // Euler formula for graphs:
        // For a closed polygon -- non-manifold mesh: T−E+V=1 -> E = T + V - 1
        // for a sphere: T−E+V=2. -> E = T + V - 2

        int edges;
        if(phi0 <= 0 && phi1 >= Pi) {
            edges = vertices.size() + (indices.size() / 3) - 2;
        }
        else if(mx::almost_equal(phi0, 0.0f) || mx::almost_equal(phi1, Pi)) {
            edges = vertices.size() + (indices.size() / 3) - 1;
        }
        else {
            edges = vertices.size() + (indices.size() / 3);
        }

        if(edges <= 0) {
            return PyTuple_Pack(2, Py_None, Py_None);
        }

        PyListObject *bonds = (PyListObject*)PyList_New(edges);

        for(int i = 0; i < vertices.size(); ++i) {
            Magnum::Vector3 pos = m.transformPoint(vertices[i]);
            MxParticleHandle *p = MxParticle_NewEx(type, pos, velocity, -1);
            parts->parts[i] = p->id;
            Py_DecRef(p);
        }

        if(vertices.size() > 0 && indices.size() == 0) {
            PyObject *result = PyTuple_New(2);
            PyTuple_SET_ITEM(result, 0, (PyObject*)parts);
            Py_INCREF(Py_None);
            PyTuple_SET_ITEM(result, 1, (PyObject*)Py_None);
            return result;
        }

        int nbonds = 0;
        for(int i = 0; i < indices.size(); i += 3) {
            int a = indices[i];
            int b = indices[i+1];
            int c = indices[i+2];

            nbonds += insert_bond(bonds, nbonds, a, b, pot, parts);
            nbonds += insert_bond(bonds, nbonds, b, c, pot, parts);
            nbonds += insert_bond(bonds, nbonds, c, a, pot, parts);
        }

        // TODO: probably excessive error message...
        if(nbonds != PyList_GET_SIZE(bonds)) {
            std::string msg = "unknown error in finding edges for sphere mesh, \n";
            msg += "vertices: " + std::to_string(vertices.size()) + "\n";
            msg += "indices: " + std::to_string(indices.size()) + "\n";
            msg += "expected edges: " + std::to_string(edges) + "\n";
            msg += "found edges: " + std::to_string(nbonds);
            throw std::overflow_error(msg);
        }

        PyObject *result = PyTuple_New(2);
        PyTuple_SET_ITEM(result, 0, (PyObject*)parts);
        PyTuple_SET_ITEM(result, 1, (PyObject*)bonds);

        // at this point, each returned object has a ref count of 1,
        // they're all either lists, or handles to stuff.
        return (PyObject*)result;
    }
    catch (const std::exception &e) {
        C_RETURN_EXP(e);
    }
}


PyObject* MxUniverse_ResetSpecies(PyObject *self, PyObject *args, PyObject *kwargs) {
    UNIVERSE_TRY();
    
    for(int i = 0; i < _Engine.s.nr_parts; ++i) {
        MxParticle *part = _Engine.s.partlist[i];
        if(part && part->state_vector) {
            part->state_vector->reset();
        }
    }
    
    for(int i = 0; i < _Engine.s.largeparts.count; ++i) {
        MxParticle *part = &_Engine.s.largeparts.parts[i];
        if(part && part->state_vector) {
            part->state_vector->reset();
        }
    }
    
    // redraw, state changed. 
    MxSimulator_Redraw();
    
    Py_RETURN_NONE;
    
    UNIVERSE_FINALLY(NULL);
}
