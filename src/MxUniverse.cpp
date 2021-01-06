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
#include <MxUniverseIterators.h>
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

static HRESULT universe_bind_potential(MxPotential *pot, PyObject *a, PyObject *b, bool bound = false);

static HRESULT universe_bind_force(MxForce *f, PyObject *a);

static PyObject *universe_virial(PyObject *_args, PyObject *_kwargs);

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
    cutoff{1},
    flags{0},
    maxTypes{64},
    dt{0.01},
    temp{1},
    nParticles{100},
    threads{mx::ThreadPool::hardwareThreadSize()},
    integrator{EngineIntegrator::FORWARD_EULER},
    boundaryConditions{PeriodicFlags::space_periodic_full},
    max_distance(-1)
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
    
    u.def_property_readonly_static("center",
           [](py::object self) -> py::handle {
               Magnum::Vector3 center = universe_center();
               PyObject *result = mx::cast(center);
               return result;
        }
    );

    u.def_static("bind", [](py::args args, py::kwargs kwargs) -> py::handle {
            UNIVERSE_CHECK();
            PyObject *result = NULL;
            PY_CHECK(MxUniverse_Bind(args.ptr(), kwargs.ptr(), &result));
            if(!result) {
                Py_RETURN_NONE;
            }
            return result;
        }
    );

    u.def_static("virial", [](py::args args, py::kwargs kwargs) -> py::handle {
            UNIVERSE_CHECK();
            return universe_virial(args.ptr(), kwargs.ptr());
        }
    );
    
    u.def_static("bind_pairwise", [](py::args args, py::kwargs kwargs) -> py::handle {
        UNIVERSE_CHECK();
        return MxPyUniverse_BindPairwise(args.ptr(), kwargs.ptr());
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
    uc.def_readwrite("cutoff", &MxUniverseConfig::cutoff);
    uc.def_readwrite("flags", &MxUniverseConfig::flags);


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
       (pl = MxParticleList_FromList(PyTuple_GetItem(args, 1)))) {
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


static HRESULT universe_bind_potential(MxPotential *p, PyObject *a, PyObject *b, bool bound) {
    MxParticleData *a_type = NULL;
    MxParticleData *b_type = NULL;
    if((a_type = MxParticleType_Get(a)) && (b_type = MxParticleType_Get(b))) {
        
        MxPotential *pot = NULL;
        
        if(p->create_func) {
            pot = p->create_func(p, (MxParticleType*)a, (MxParticleType*)b);
        }
        else {
            pot = p;
        }
        
        if(bound) {
            pot->flags = pot->flags | POTENTIAL_BOUND;
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
        MxParticleHandle *a_part = ((MxParticleHandle *)a);
        MxParticleHandle *b_part = ((MxParticleHandle *)b);

        //MxBond_New(uint32_t flags,
        //        int32_t i, int32_t j,
        //        double half_life,
        //        double bond_energy,
        //        struct MxPotential* potential);

        MxBondHandle_New(0, a_part->id, b_part->id,
                std::numeric_limits<double>::max(),
                std::numeric_limits<double>::max(),
                p);

        return S_OK;
    }

    return mx_error(E_FAIL, "can only add potential to particle types or instances");
}

static HRESULT universe_bind_force(MxForce *f, PyObject *a) {
    MxParticleData *a_type = MxParticleType_Get(a);
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

static void universe_step(py::args args, py::kwargs kwargs) {

    double until = arg<double>("until", 0, args.ptr(), kwargs.ptr());
    double dt = arg<double>("dt", 1, args.ptr(), kwargs.ptr());

    PY_CHECK(MxUniverse_Step(until, dt));
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


PyObject *universe_virial(PyObject *_args, PyObject *_kwargs) {
    try {
        PyObject *_origin = mx::arg("origin", 0, _args, _kwargs);
        PyObject *_radius = mx::arg("radius", 1, _args, _kwargs);
        PyObject *_types = mx::arg("types", 2, _args, _kwargs);
        
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
        c_exp(e, "error checking args");
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
        PyObject *ppot = mx::arg(names[0], 0, args, kwds);
        PyObject *pparts = mx::arg(names[1], 1, args, kwds);
        PyObject *pcutoff = mx::arg(names[2], 2, args, kwds);
        PyObject *pairs = mx::arg(names[3], 3, args, kwds);
        
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
        
        if((parts = MxParticleList_FromList(pparts)) == NULL) {
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
        c_exp(e, "error");
        return NULL;
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
        PyObject *ppot = mx::arg("potential", 0, args, kwds);
        PyObject *pn = mx::arg("n", 1, args, kwds);
        PyObject *pcenter = mx::arg("center", 2, args, kwds);
        PyObject *pradius = mx::arg("radius", 3, args, kwds);
        PyObject *pphi = mx::arg("phi", 4, args, kwds);
        PyObject *type = mx::arg("type", 5, args, kwds);
        
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
        c_exp(e, "error");
        return NULL;
    }
}
