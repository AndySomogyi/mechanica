/*
 * MxUniverse.cpp
 *
 *  Created on: Sep 7, 2019
 *      Author: andy
 */

#include <MxUniverse.h>
#include <MxUniverseIterators.h>
#include <iostream>
#include <MxForce.h>
#include <MxPy.h>
#include <MxSimulator.h>
#include <limits>

#define PY_CHECK(hr) {if(!SUCCEEDED(hr)) { throw py::error_already_set();}}

static void print_performance_counters();

namespace py = pybind11;

using Magnum::Vector3;

MxUniverse Universe = {
    .isRunning = false,
    .performance_info_display_interval = 100,
    .performance_info_flags =
        ENGINE_TIMER_STEP |
        ENGINE_TIMER_NONBOND |
        ENGINE_TIMER_BONDED |
        ENGINE_TIMER_ADVANCE
};

static HRESULT universe_bind_potential(MxPotential *pot, PyObject *a, PyObject *b);

static HRESULT universe_bind_force(MxForce *f, PyObject *a);

// the single static engine instance per process

// complete and total hack to get the global engine to show up here
// instead of the mdcore static lib.
// TODO: fix this crap.
engine _Engine = {
        .flags = 0
};

// default to running universe.
static uint32_t universe_flags = MxUniverse_Flags::MX_RUNNING;


CAPI_FUNC(struct engine*) engine_get()
{
    return &_Engine;
}

#define UNIVERSE_CHECK() { \
    if (_Engine.flags == 0 ) { \
        std::string err = "Error in "; \
        err += MX_FUNCTION; \
        err += ", Universe not initialized"; \
        throw std::domain_error(err.c_str()); \
    } \
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

MxUniverseConfig::MxUniverseConfig() :
    origin {0, 0, 0},
    dim {10, 10, 10},
    spaceGridSize {4, 4, 4},
    boundaryConditions{1, 1, 1},
    cutoff{1},
    flags{0},
    maxTypes{64},
    dt{0.01},
    temp{1},
    nParticles{100},
    threads{4}
{
}

/**
 *  basic wrapper for universe functions.
 *
 *  Everything is static in the universe.
 */
struct PyUniverse {
};


static Vector3 universe_dim(py::object /* self */) {
    UNIVERSE_CHECK();
    return Magnum::Vector3{(float)_Engine.s.dim[0], (float)_Engine.s.dim[1], (float)_Engine.s.dim[2]};
}

static PyUniverse *py_universe_init(const MxUniverseConfig &conf) {
    if(_Engine.flags) {
        throw std::domain_error("Error, Universe is already initialized");
    }


    MxUniverse_Init(conf);


    return new PyUniverse();
}




PyTypeObject MxUniverse_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name =           "Universe",
    .tp_basicsize =      sizeof(MxUniverse),
    .tp_itemsize =       0, 
    .tp_dealloc =        0, 
    .tp_print =          0, 
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
    .tp_methods =        0, 
    .tp_members =        0, 
    .tp_getset =         0, 
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
    py::class_<PyUniverse> u(m, "Universe");
    u.def(py::init(&py_universe_init));
     //sim.def(py::init(&PySimulator_New), py::return_value_policy::reference);
     //sim.def_property_readonly("foo", &PySimulator::foo);
     //sim.def_static("poll_events", [](){PY_CHECK(MxSimulator_PollEvents());});
     //sim.def_static("wait_events", &pysimulator_wait_events);
     //sim.def_static("post_empty_event", [](){PY_CHECK(MxSimulator_PostEmptyEvent());});

    u.def_property_readonly_static("dim", &universe_dim);

    u.def_property_readonly_static("temperature",
        [](py::object self) -> double {
            UNIVERSE_CHECK();
            return engine_temperature(&_Engine);
        }
    );

    u.def_property_readonly_static("kinetic_energy",
        [](py::object self) -> double {
            UNIVERSE_CHECK();
            return engine_kinetic_energy(&_Engine);
        }
    );

    u.def_property_readonly_static("time",
        [](py::object self) -> double {
            UNIVERSE_CHECK();
            return _Engine.time * _Engine.dt;
        }
    );

    u.def_property_readonly_static("dt",
        [](py::object self) -> double {
            UNIVERSE_CHECK();
            return _Engine.dt;
        }
    );

    u.def_property_readonly_static("on_time",
            [](py::object self) -> py::handle {
                UNIVERSE_CHECK();
                return (PyObject*)_Engine.on_time;
        }
    );

    u.def_property_readonly_static("particles",
            [](py::object self) -> py::handle {
                static PyParticles particles;
                return py::cast(particles).release();
            }
        );
    
    u.def_property_readonly_static("bonds",
            [](py::object self) -> py::handle {
                static PyBonds bonds;
                return py::cast(bonds).release();
            }
        );

    u.def_static("bind", [](py::args args, py::kwargs kwargs) -> void {
            UNIVERSE_CHECK();
            PY_CHECK(MxUniverse_Bind(args.ptr(), kwargs.ptr()));
            return;
        }
    );

    u.def_static("start", []() -> void {
            UNIVERSE_CHECK();
            PY_CHECK(MxUniverse_SetFlag(MxUniverse_Flags::MX_RUNNING, true));
            return;
        }
    );

    u.def_static("stop", []() -> void {
            PY_CHECK(MxUniverse_SetFlag(MxUniverse_Flags::MX_RUNNING, false));
            return;
        }
    );

    u.def_static("step", [](py::args args, py::kwargs kwargs) -> void {
        double until = arg<double>("until", 0, args.ptr(), kwargs.ptr(), 0);
        double dt = arg<double>("dt", 1, args.ptr(), kwargs.ptr(), 0);
        PY_CHECK(MxUniverse_Step(until, dt));
    });



    py::class_<MxUniverseConfig> uc(u, "Config");
    uc.def(py::init());
    uc.def_readwrite("origin", &MxUniverseConfig::origin);
    uc.def_readwrite("dim", &MxUniverseConfig::dim);
    uc.def_readwrite("space_grid_size", &MxUniverseConfig::spaceGridSize);
    uc.def_readwrite("boundary_conditions", &MxUniverseConfig::boundaryConditions);
    uc.def_readwrite("cutoff", &MxUniverseConfig::cutoff);
    uc.def_readwrite("flags", &MxUniverseConfig::flags);


    return S_OK;
}


CAPI_FUNC(HRESULT) MxUniverse_Bind(PyObject *args, PyObject *kwargs)
{
    if(args && PyTuple_Size(args) == 3) {
        return MxUniverse_BindThing2(PyTuple_GetItem(args, 0), PyTuple_GetItem(args, 1), PyTuple_GetItem(args, 2));
    }
    
    if(args && PyTuple_Size(args) == 2) {
        return MxUniverse_BindThing1(PyTuple_GetItem(args, 0), PyTuple_GetItem(args, 1));
    }
    
    return mx_error(E_FAIL, "bind only implmented for 2 or 3 arguments: bind(thing, a, b)");
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


static HRESULT universe_bind_potential(MxPotential *p, PyObject *a, PyObject *b) {
    if(PyObject_TypeCheck(a, &MxParticleType_Type) &&
       PyObject_TypeCheck(b, &MxParticleType_Type)) {
        MxParticleData *a_type = ((MxParticleType *)a);
        MxParticleData *b_type = ((MxParticleType *)b);
        
        MxPotential *pot = NULL;
        
        if(p->create_func) {
            pot = p->create_func(p, (MxParticleType*)a, (MxParticleType*)b);
        }
        else {
            pot = p;
        }
        
        if(engine_addpot(&_Engine, pot, a_type->id, b_type->id) != engine_err_ok) {
            std::string msg = "failed to add potential to engine: error";
            msg += std::to_string(engine_err);
            msg += ", ";
            msg += engine_err_msg[-engine_err];
            return mx_error(E_FAIL, msg.c_str());
        }
        return S_OK;
    }

    if(MxParticle_Check(a) && MxParticle_Check(b)) {
        MxPyParticle *a_part = ((MxPyParticle *)a);
        MxPyParticle *b_part = ((MxPyParticle *)b);

        //MxBond_New(uint32_t flags,
        //        int32_t i, int32_t j,
        //        double half_life,
        //        double bond_energy,
        //        struct MxPotential* potential);

        MxBond_New(0, a_part->id, b_part->id,
                std::numeric_limits<double>::max(),
                std::numeric_limits<double>::max(),
                p);

        return S_OK;
    }

    return mx_error(E_FAIL, "can only add potential to particle types or instances");
}

static HRESULT universe_bind_force(MxForce *f, PyObject *a) {
    if(PyObject_TypeCheck(a, &MxParticleType_Type)) {
        MxParticleData *a_type = ((MxParticleType *)a);
        
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

static void universe_step(py::args args, py::kwargs kwargs) {

    double until = arg<double>("until", 0, args.ptr(), kwargs.ptr());
    double dt = arg<double>("dt", 1, args.ptr(), kwargs.ptr());

    PY_CHECK(MxUniverse_Step(until, dt));
}


CAPI_FUNC(HRESULT) MxUniverse_Step(double until, double dt) {
    
    ticks tic, toc_step, toc_temp;
    
    double epot, ekin, v2, temp;
    
    int   k, cid, pid;
    
    double w;
    
    // take a step
    tic = getticks();
    
    if ( engine_step( &_Engine ) != 0 ) {
        printf("main: engine_step failed with engine_err=%i.\n",engine_err);
        errs_dump(stdout);
        // TODO: correct error reporting
        return E_FAIL;
    }
    
    toc_step = getticks();
    
    
    toc_temp = getticks();

    MxSimulator_Redraw();
    
    if(universe_flags & MxUniverse_Flags::MX_SHOW_PERF_STATS ) {
        print_performance_counters();
    }

    return S_OK;
}


CAPI_FUNC(HRESULT) MxUniverse_Init(const MxUniverseConfig &conf) {
    double origin[3] = {conf.origin[0], conf.origin[1], conf.origin[2]};
    double dim[3] = {conf.dim[0], conf.dim[1], conf.dim[2]};
    double L[3] = {conf.dim[0] / conf.spaceGridSize[0], conf.dim[1] / conf.spaceGridSize[1], conf.dim[2] / conf.spaceGridSize[2]};



    int er = engine_init ( &_Engine , origin , dim , L ,
            conf.cutoff, space_periodic_full , conf.maxTypes , conf.flags );

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
    if(_Engine.time % Universe.performance_info_display_interval) {
        return;
    }
    
    std::cout << "performance_timers : { " << std::endl;
    std::cout << "\t engine_step: " << ms(_Engine.timers[engine_timer_step]) << std::endl;
    std::cout << "\t engine_nonbond: " << ms(_Engine.timers[engine_timer_nonbond]) << std::endl;
    std::cout << "\t engine_bonded: " << ms(_Engine.timers[engine_timer_bonded]) << std::endl;
    std::cout << "\t engine_advance: " << ms(_Engine.timers[engine_timer_advance]) << std::endl;
    std::cout << "}" << std::endl;
}
