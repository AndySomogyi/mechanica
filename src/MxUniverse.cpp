/*
 * MxUniverse.cpp
 *
 *  Created on: Sep 7, 2019
 *      Author: andy
 */

#include <MxUniverse.h>
#include <MxUniverseIterators.h>
#include <iostream>


#include <MxPy.h>

#define PY_CHECK(hr) {if(!SUCCEEDED(hr)) { throw py::error_already_set();}}

namespace py = pybind11;

using Magnum::Vector3;

MxUniverse Universe;

static HRESULT universe_bind_potential(MxPotential *pot, PyObject *a, PyObject *b);

// the single static engine instance per process

// complete and total hack to get the global engine to show up here
// instead of the mdcore static lib.
// TODO: fix this crap.
engine _Engine = {
        .flags = 0
};

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

MxUniverseConfig::MxUniverseConfig() :
    origin {0, 0, 0},
    dim {10, 10, 10},
    spaceGridSize {2, 2, 2},
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
    return Magnum::Vector3{_Engine.s.dim[0], _Engine.s.dim[1], _Engine.s.dim[2]};
}

static PyUniverse *universe_init(const MxUniverseConfig &conf) {
    if(_Engine.flags) {
        throw std::domain_error("Error, Universe is already initialized");
    }

    double origin[3] = {conf.origin[0], conf.origin[1], conf.origin[2]};
    double dim[3] = {conf.dim[0], conf.dim[1], conf.dim[2]};
    double L[3] = {conf.dim[0] / conf.spaceGridSize[0], conf.dim[1] / conf.spaceGridSize[1], conf.dim[2] / conf.spaceGridSize[2]};



    int er = engine_init ( &_Engine , origin , dim , L ,
            conf.cutoff, space_periodic_full , conf.maxTypes , conf.flags );

    //Universe = new MxUniverse();


    return new PyUniverse();
}




PyTypeObject MxUniverse_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "Universe",
    .tp_basicsize = sizeof(MxUniverse),
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
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, 
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
    return Vector3{_Engine.s.origin[0], _Engine.s.origin[1], _Engine.s.origin[2]};
}

Magnum::Vector3 MxUniverse::dim()
{
    return Vector3{_Engine.s.dim[0], _Engine.s.dim[1], _Engine.s.dim[2]};
}



HRESULT _MxUniverse_init(PyObject* m)
{
    py::class_<PyUniverse> u(m, "Universe");
    u.def(py::init(&universe_init));
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

    u.def_property_readonly_static("particles",
            [](py::object self) -> py::handle {
                static PyParticles particles;
                return py::cast(particles).release();
            }
        );
    
    u.def_static("bind", [](py::args args, py::kwargs kwargs) -> void {
            UNIVERSE_CHECK();
            PY_CHECK(MxUniverse_Bind(args.ptr(), kwargs.ptr()));
            return;
        }
    );



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

HRESULT MxUniverse::init(const struct MxUniverseConfig &conf)
{
}

CAPI_FUNC(HRESULT) MxUniverse_Bind(PyObject *args, PyObject *kwargs)
{
    if(args && PyTuple_Size(args) == 3) {
        return MxUniverse_BindThing2(PyTuple_GetItem(args, 0), PyTuple_GetItem(args, 1), PyTuple_GetItem(args, 2));
    }
    return mx_error(E_FAIL, "bind only implmented for 3 arguments: bind(thing, a, b)");
}

CAPI_FUNC(HRESULT) MxUniverse_BindThing2(PyObject *thing, PyObject *a,
        PyObject *b)
{
    if(PyObject_IsInstance(thing, (PyObject*)&MxPotential_Type)) {
        return universe_bind_potential((MxPotential*)thing, a, b);
    }
    return mx_error(E_NOTIMPL, "binding currently implmented for potentials to things");
}


static HRESULT universe_bind_potential(MxPotential *pot, PyObject *a, PyObject *b) {
    if(PyObject_TypeCheck(a, &MxParticleType_Type) &&
       PyObject_TypeCheck(b, &MxParticleType_Type)) {
        MxParticleData *a_type = ((MxParticleType *)a)->data;
        MxParticleData *b_type = ((MxParticleType *)b)->data;
        
        if(engine_addpot(&_Engine, pot, a_type->id, b_type->id) != engine_err_ok) {
            std::string msg = "failed to add potential to engine: error";
            msg += std::to_string(engine_err);
            msg += ", ";
            msg += engine_err_msg[-engine_err];
            return mx_error(E_FAIL, msg.c_str());
        }
        return S_OK;
    }
    return mx_error(E_FAIL, "can only add potential to particle types");
}
