/*
 * MxSimulator.cpp
 *
 *  Created on: Feb 1, 2017
 *      Author: andy
 */

#include <MxSimulator.h>
#include <rendering/MxUI.h>
#include <rendering/MxTestView.h>

#include <Magnum/GL/Context.h>

#include "rendering/MxApplication.h"
#include "rendering/MxUniverseRenderer.h"
#include <rendering/MxGlfwApplication.h>
#include <rendering/MxWindowlessApplication.h>
#include <map>
#include <sstream>
#include <MxUniverse.h>
#include <MxConvert.hpp>

// mdcore errs.h
#include <errs.h>


#include <pybind11/pybind11.h>
namespace py = pybind11;


static std::vector<Vector3> fillCubeRandom(const Vector3 &corner1, const Vector3 &corner2, int nParticles);

/* What to do if ENGINE_FLAGS was not defined? */
#ifndef ENGINE_FLAGS
#define ENGINE_FLAGS engine_flag_none
#endif
#ifndef CPU_TPS
#define CPU_TPS 2.67e+9
#endif

static MxSimulator* Simulator = NULL;

static void simulator_interactive_run();

static void ipythonInputHook(py::args args);


static PyObject *context_is_current(PyObject *args, PyObject *kwargs) {
    try {
        
    }
    catch(const std::exception &e) {
        C_RETURN_EXP(e);
    }
}

static PyObject *context_make_current(PyObject *args, PyObject *kwargs) {
    try {
        
    }
    catch(const std::exception &e) {
        C_RETURN_EXP(e);
    }
}

static PyObject *context_release(PyObject *args, PyObject *kwargs) {
    try {
        
    }
    catch(const std::exception &e) {
        C_RETURN_EXP(e);
    }
}

static PyObject *camera_rotate(PyObject *args, PyObject *kwargs) {

        MxSimulator *sim = MxSimulator::Get();
    
    
        
        MxUniverseRenderer *renderer = sim->app->getRenderer();
        
        Magnum::Mechanica::ArcBall *ab = renderer->_arcball;
        
        Magnum::Vector3 eulerAngles = mx::arg<Magnum::Vector3>("euler_angles", 0, args, kwargs);
        
        ab->rotate(eulerAngles);
        
        Py_RETURN_NONE;
}


MxSimulator::Config::Config():
            _title{"Mechanica Application"},
            _size{800, 600},
            _dpiScalingPolicy{DpiScalingPolicy::Default},
            queues{4},
           _windowless{ false }
{
    _windowFlags = MxSimulator::WindowFlags::Resizable | 
                   MxSimulator::WindowFlags::Focused   | 
                   MxSimulator::WindowFlags::Hidden;  // make the window initially hidden 
}



MxSimulator::GLConfig::GLConfig():
_colorBufferSize{8, 8, 8, 0}, _depthBufferSize{24}, _stencilBufferSize{0},
_sampleCount{0}, _version{GL::Version::None},
#ifndef MAGNUM_TARGET_GLES
_flags{Flag::ForwardCompatible},
#else
_flags{},
#endif
_srgbCapable{false} {}

MxSimulator::GLConfig::~GLConfig() = default;




struct Foo {
    Foo(const std::string &name) : name(name) { }
    void setName(const std::string &name_) { name = name_; }
    const std::string &getName() const { return name; }
    
    void stuff(py::args args, py::kwargs kwargs) {
        
        
        std::cout << "hi" << std::endl;
    }

    std::string name;

    Magnum::Vector3 vec;
};






#define SIMULATOR_CHECK()  if (!Simulator) { return mx_error(E_INVALIDARG, "Simulator is not initialized"); }

#define PY_CHECK(hr) {if(!SUCCEEDED(hr)) { throw py::error_already_set();}}

#define PYSIMULATOR_CHECK() { \
    if(!Simulator) { \
        throw std::domain_error(std::string("Simulator Error in ") + MX_FUNCTION + ": Simulator not initialized"); \
    } \
}




void test(const Foo& f) {
    std::cout << "hello from test" << f.name << std::endl;
}

Foo *make_foo(py::str ps) {
    std::string s = py::cast<std::string>(ps);
    return new Foo(s);
}




void foo(PyObject *m) {
    py::class_<Foo> c(m, "Foo");
    
    
        c.def(py::init<const std::string &>());
        c.def(py::init(&make_foo));
        c.def("setName", &Foo::setName);
        c.def("getName", &Foo::getName);
    
        c.def("stuff", &Foo::stuff);
    
    c.def_readwrite("vec", &Foo::vec);

    py::implicitly_convertible<py::str, Foo>();

        PyObject *p = c.ptr();

        py::module mm = py::reinterpret_borrow<py::module>(m);

        mm.def("test", &test);
    
    std::cout << "name: " << p->ob_type->tp_name << std::endl;
}


/**
 * Make a Arguments struct from a python string list,
 * Agh!!! Magnum has different args for different app types,
 * so this needs to be a damned template.
 */
template<typename T>
struct ArgumentsWrapper  {

    ArgumentsWrapper(py::list args) {

        for(auto o : args) {
            strings.push_back(o.cast<std::string>());
            cstrings.push_back(strings.back().c_str());
            
            std::cout << "args: " << cstrings.back() << std::endl;
        }
        
        // stupid thing is a int reference, keep an ivar around for it
        // to point to. 
        argsSeriouslyTakesAFuckingIntReference = cstrings.size();
        char** fuckingConstBullshit = const_cast<char**>(cstrings.data());
        
        pArgs = new T(argsSeriouslyTakesAFuckingIntReference, fuckingConstBullshit);
    }
    
    ~ArgumentsWrapper() {
        delete pArgs;
    }


    // OMG this is a horrible design.
    // how I hate C++
    std::vector<std::string> strings;
    std::vector<const char*> cstrings;
    T *pArgs = NULL;
    int argsSeriouslyTakesAFuckingIntReference;
};


static void parse_kwargs(PyObject *kwargs, MxSimulator::Config &conf) {
    
    PyObject *o;
        
    if((o = PyDict_GetItemString(kwargs, "dim"))) {
        conf.universeConfig.dim = mx::cast<Magnum::Vector3>(o);
    }
    
    if((o = PyDict_GetItemString(kwargs, "cutoff"))) {
        conf.universeConfig.cutoff = mx::cast<double>(o);
    }
    
    if((o = PyDict_GetItemString(kwargs, "cells"))) {
        conf.universeConfig.spaceGridSize = mx::cast<Vector3i>(o);
    }
    
    if((o = PyDict_GetItemString(kwargs, "threads"))) {
        conf.universeConfig.threads = mx::cast<unsigned>(o);
    }

    if((o = PyDict_GetItemString(kwargs, "integrator"))) {
        conf.universeConfig.integrator = py::cast<EngineIntegrator>(o);
    }

    if((o = PyDict_GetItemString(kwargs, "dt"))) {
        conf.universeConfig.dt = mx::cast<double>(o);
    }
    
    if((o = PyDict_GetItemString(kwargs, "bc"))) {
        conf.universeConfig.setBoundaryConditions(o);
    }
    
    if((o = PyDict_GetItemString(kwargs, "boundary_conditions"))) {
        conf.universeConfig.setBoundaryConditions(o);
    }
    
    if((o = PyDict_GetItemString(kwargs, "max_distance"))) {
        conf.universeConfig.max_distance = mx::cast<double>(o);
    }
    
    if((o = PyDict_GetItemString(kwargs, "windowless"))) {
        conf.setWindowless(mx::cast<bool>(o));
        if(conf.windowless()) {
            conf.setWindowSize({1024,1024});
        }
    }
    
    if((o = PyDict_GetItemString(kwargs, "window_size"))) {
        Magnum::Vector2i windowSize = mx::cast<Magnum::Vector2i>(o);
        conf.setWindowSize(windowSize);
    }
    
    if((o = PyDict_GetItemString(kwargs, "perfcounters"))) {
        conf.universeConfig.timers_mask = mx::cast<uint32_t>(o);
    }
    
    if((o = PyDict_GetItemString(kwargs, "perfcounter_period"))) {
        conf.universeConfig.timer_output_period = mx::cast<int>(o);
    }
}

static HRESULT simulator_init(py::args args, py::kwargs kwargs);

/**
 * Create a private 'python' flavored version of the simulator
 * interface here to wrap with pybind11.
 *
 * Don't want to introdude pybind header file into main project and
 * and polute other files.
 */
struct PySimulator  {
    
    PySimulator(py::args args, py::kwargs kwargs) {
        PY_CHECK(simulator_init(args, kwargs));
    }
    
    py::handle foo() {
        std::cout << MX_FUNCTION << std::endl;
        PyObject *o = PyLong_FromLong(3);
        
        py::handle h(o);
        
        return h;
    };
    
    ~PySimulator() {
    }
};


static std::string gl_info(const Magnum::Utility::Arguments &args);


static PyObject *not_initialized_error();



// (5) Initializer list constructor
const std::map<std::string, int> configItemMap {
    {"none", MXSIMULATOR_NONE},
    {"windowless", MXSIMULATOR_WINDOWLESS},
    {"glfw", MXSIMULATOR_GLFW}
};


/**
 * tp_alloc(type) to allocate storage
 * tp_new(type, args) to create blank object
 * tp_init(obj, args) to initialize object
 */

static int init(PyObject *self, PyObject *args, PyObject *kwds)
{
    std::cout << MX_FUNCTION << std::endl;

    MxSimulator *s = new (self) MxSimulator();
    return 0;
}



static PyObject *simulator_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    return NULL;
}


#define MX_CLASS METH_CLASS | METH_VARARGS | METH_KEYWORDS



static PyMethodDef methods[] = {
        { "pollEvents", (PyCFunction)MxPyUI_PollEvents, MX_CLASS, NULL },
        { "waitEvents", (PyCFunction)MxPyUI_WaitEvents, MX_CLASS, NULL },
        { "postEmptyEvent", (PyCFunction)MxPyUI_PostEmptyEvent, MX_CLASS, NULL },
        { "initializeGraphics", (PyCFunction)MxPyUI_InitializeGraphics, MX_CLASS, NULL },
        { "createTestWindow", (PyCFunction)MxPyUI_CreateTestWindow, MX_CLASS, NULL },
        { "testWin", (PyCFunction)PyTestWin, MX_CLASS, NULL },
        { "destroyTestWindow", (PyCFunction)MxPyUI_DestroyTestWindow, MX_CLASS, NULL },
        { NULL, NULL, 0, NULL }
};



// TODO: this is a total hack, mostly because I don't fully understand pybind
//       FIX THIS SHIT!
PySimulator *PySimulator_New(py::args args, py::kwargs kwargs) {
    return (PySimulator*)new PySimulator(args, kwargs);
};


static void pysimulator_wait_events(py::args args) {
    if(args.size() == 0) {
        PY_CHECK(MxSimulator_WaitEvents());
    }
    else if(args.size() == 1) {
        double t = args[0].cast<double>();
        PY_CHECK(MxSimulator_WaitEventsTimeout(t));
    }
    else {
        mx_error(E_INVALIDARG, "wait_events only only accepts 0 or 1 arguments");
        PY_CHECK(E_FAIL);
    }
};

static py::object ftest() {
    return py::cpp_function([](int x) -> int {return x + 10;});
}

HRESULT _MxSimulator_init(PyObject* m) {

    std::cout << MX_FUNCTION << std::endl;
    
    py::class_<PySimulator> sim(m, "Simulator");
    sim.def(py::init(&PySimulator_New), py::return_value_policy::reference);
    sim.def_property_readonly("foo", &PySimulator::foo);
    sim.def_static("poll_events", [](){PY_CHECK(MxSimulator_PollEvents());});
    sim.def_static("wait_events", &pysimulator_wait_events);
    sim.def_static("post_empty_event", [](){PY_CHECK(MxSimulator_PostEmptyEvent());});
    sim.def_static("run", [](){PY_CHECK(MxSimulator_Run(-1));});
    sim.def_static("run", [](double et){PY_CHECK(MxSimulator_Run(et));});
    sim.def_static("ftest", &ftest);
    sim.def_static("irun", [] () { PY_CHECK(MxSimulator_InteractiveRun()); });
    sim.def_static("show", [] () { PY_CHECK(MxSimulator_Show()); });
    sim.def_static("close", [] () { PY_CHECK(MxSimulator_Close()); });
    
    sim.def_static("context_is_current", [](py::args args, py::kwargs kwargs) -> py::handle {
        return context_is_current(args.ptr(), kwargs.ptr());
    });
    sim.def_static("context_make_current", [](py::args args, py::kwargs kwargs) -> py::handle {
        return context_make_current(args.ptr(), kwargs.ptr());
    });
    sim.def_static("context_release", [](py::args args, py::kwargs kwargs) -> py::handle {
        return context_release(args.ptr(), kwargs.ptr());
    });
    sim.def_static("camera_rotate", [](py::args args, py::kwargs kwargs) -> py::handle {
        return camera_rotate(args.ptr(), kwargs.ptr());
    });
    
    sim.def_property_readonly_static("threads", [] (py::object) -> int {
        PYSIMULATOR_CHECK();
        return _Engine.nr_runners;
    });
    
    sim.def_property_readonly_static("window", [](py::object) -> py::handle {
            PYSIMULATOR_CHECK();
            return py::handle(Simulator->app->getWindow());
        }
    );

    py::enum_<MxSimulator::WindowFlags>(sim, "WindowFlags", py::arithmetic())
            .value("Fullscreen", MxSimulator::WindowFlags::Fullscreen)
            .value("Resizable", MxSimulator::WindowFlags::Resizable)
            .value("Hidden", MxSimulator::WindowFlags::Hidden)
            .value("Maximized", MxSimulator::WindowFlags::Maximized)
            .value("Minimized", MxSimulator::WindowFlags::Minimized)
            .value("AlwaysOnTop", MxSimulator::WindowFlags::AlwaysOnTop)
            .value("AutoIconify", MxSimulator::WindowFlags::AutoIconify)
            .value("Focused", MxSimulator::WindowFlags::Focused)
            .value("Contextless", MxSimulator::WindowFlags::Contextless)
            .export_values();

    py::enum_<EngineIntegrator>(m, "Integrator")
            .value("FORWARD_EULER", EngineIntegrator::FORWARD_EULER)
            .value("RUNGE_KUTTA_4", EngineIntegrator::RUNGE_KUTTA_4)
            .export_values();

 

    py::class_<MxSimulator::Config> sc(sim, "Config");
    sc.def(py::init());
    sc.def_property("window_title", &MxSimulator::Config::title, &MxSimulator::Config::setTitle);
    sc.def_property("window_size", &MxSimulator::Config::windowSize, &MxSimulator::Config::setWindowSize);
    sc.def_property("dpi_scaling", &MxSimulator::Config::dpiScaling, &MxSimulator::Config::setDpiScaling);
    sc.def_property("window_flags", &MxSimulator::Config::windowFlags, &MxSimulator::Config::setWindowFlags);
    sc.def_property("windowless", &MxSimulator::Config::windowless, &MxSimulator::Config::setWindowless);
    
    sc.def_property("size", &MxSimulator::Config::size, &MxSimulator::Config::setSize);
    
    sc.def_property("dim", [](const MxSimulator::Config &conf) { return conf.universeConfig.dim; },
                    [](MxSimulator::Config &conf, Magnum::Vector3& vec) {conf.universeConfig.dim = vec;});
    
    sc.def_property("space_grid_size", [](const MxSimulator::Config &conf) { return conf.universeConfig.spaceGridSize; },
                    [](MxSimulator::Config &conf, Magnum::Vector3i& vec) {conf.universeConfig.spaceGridSize = vec;});
    
    sc.def_property("cutoff", [](const MxSimulator::Config &conf) { return conf.universeConfig.cutoff; },
                    [](MxSimulator::Config &conf, double v) {conf.universeConfig.cutoff = v;});
    
    sc.def_property("flags", [](const MxSimulator::Config &conf) { return conf.universeConfig.flags; },
                    [](MxSimulator::Config &conf, uint32_t v) {conf.universeConfig.flags = v;});
    
    sc.def_property("max_types", [](const MxSimulator::Config &conf) { return conf.universeConfig.maxTypes; },
                    [](MxSimulator::Config &conf, uint32_t v) {conf.universeConfig.maxTypes = v;});
    
    sc.def_property("dt", [](const MxSimulator::Config &conf) { return conf.universeConfig.dt; },
                    [](MxSimulator::Config &conf, double v) {conf.universeConfig.dt = v;});
    
    sc.def_property("temp", [](const MxSimulator::Config &conf) { return conf.universeConfig.temp; },
                    [](MxSimulator::Config &conf, double v) {conf.universeConfig.temp = v;});
    
    sc.def_property("threads", [](const MxSimulator::Config &conf) { return conf.universeConfig.threads; },
                    [](MxSimulator::Config &conf, int v) {conf.universeConfig.threads = v;});
    
    sc.def_property("integrator", [](const MxSimulator::Config &conf) { return conf.universeConfig.integrator; },
                    [](MxSimulator::Config &conf, EngineIntegrator v) {conf.universeConfig.integrator = v;});


    py::class_<MxSimulator::GLConfig> gc(sim, "GLConfig");
    gc.def(py::init());
    gc.def_property("color_buffer_size", &MxSimulator::GLConfig::colorBufferSize, &MxSimulator::GLConfig::setColorBufferSize);
    gc.def_property("depth_buffer_size", &MxSimulator::GLConfig::depthBufferSize, &MxSimulator::GLConfig::setDepthBufferSize);
    gc.def_property("stencil_buffer_size", &MxSimulator::GLConfig::stencilBufferSize, &MxSimulator::GLConfig::setStencilBufferSize);
    gc.def_property("sample_count", &MxSimulator::GLConfig::sampleCount, &MxSimulator::GLConfig::setSampleCount);
    gc.def_property("srgb_capable", &MxSimulator::GLConfig::isSrgbCapable, &MxSimulator::GLConfig::setSrgbCapable);


    py::enum_<MxWindowAttributes>(m, "WindowAttributes", py::arithmetic())
        .value("FOCUSED", MxWindowAttributes::MX_FOCUSED)
        .value("ICONIFIED", MxWindowAttributes::MX_ICONIFIED)
        .value("RESIZABLE", MxWindowAttributes::MX_RESIZABLE)
        .value("VISIBLE", MxWindowAttributes::MX_VISIBLE)
        .value("DECORATED", MxWindowAttributes::MX_DECORATED)
        .value("AUTO_ICONIFY", MxWindowAttributes::MX_AUTO_ICONIFY)
        .value("FLOATING", MxWindowAttributes::MX_FLOATING)
        .value("MAXIMIZED", MxWindowAttributes::MX_MAXIMIZED)
        .value("CENTER_CURSOR", MxWindowAttributes::MX_CENTER_CURSOR)
        .value("TRANSPARENT_FRAMEBUFFER", MxWindowAttributes::MX_TRANSPARENT_FRAMEBUFFER)
        .value("HOVERED", MxWindowAttributes::MX_HOVERED)
        .value("FOCUS_ON_SHOW", MxWindowAttributes::MX_FOCUS_ON_SHOW)
        .value("RED_BITS", MxWindowAttributes::MX_RED_BITS)
        .value("GREEN_BITS", MxWindowAttributes::MX_GREEN_BITS)
        .value("BLUE_BITS", MxWindowAttributes::MX_BLUE_BITS)
        .value("ALPHA_BITS", MxWindowAttributes::MX_ALPHA_BITS)
        .value("DEPTH_BITS", MxWindowAttributes::MX_DEPTH_BITS)
        .value("STENCIL_BITS", MxWindowAttributes::MX_STENCIL_BITS)
        .value("ACCUM_RED_BITS", MxWindowAttributes::MX_ACCUM_RED_BITS)
        .value("ACCUM_GREEN_BITS", MxWindowAttributes::MX_ACCUM_GREEN_BITS)
        .value("ACCUM_BLUE_BITS", MxWindowAttributes::MX_ACCUM_BLUE_BITS)
        .value("ACCUM_ALPHA_BITS", MxWindowAttributes::MX_ACCUM_ALPHA_BITS)
        .value("AUX_BUFFERS", MxWindowAttributes::MX_AUX_BUFFERS)
        .value("STEREO", MxWindowAttributes::MX_STEREO)
        .value("SAMPLES", MxWindowAttributes::MX_SAMPLES)
        .value("SRGB_CAPABLE", MxWindowAttributes::MX_SRGB_CAPABLE)
        .value("REFRESH_RATE", MxWindowAttributes::MX_REFRESH_RATE)
        .value("DOUBLEBUFFER", MxWindowAttributes::MX_DOUBLEBUFFER)
        .value("CLIENT_API", MxWindowAttributes::MX_CLIENT_API)
        .value("CONTEXT_VERSION_MAJOR", MxWindowAttributes::MX_CONTEXT_VERSION_MAJOR)
        .value("CONTEXT_VERSION_MINOR", MxWindowAttributes::MX_CONTEXT_VERSION_MINOR)
        .value("CONTEXT_REVISION", MxWindowAttributes::MX_CONTEXT_REVISION)
        .value("CONTEXT_ROBUSTNESS", MxWindowAttributes::MX_CONTEXT_ROBUSTNESS)
        .value("OPENGL_FORWARD_COMPAT", MxWindowAttributes::MX_OPENGL_FORWARD_COMPAT)
        .value("OPENGL_DEBUG_CONTEXT", MxWindowAttributes::MX_OPENGL_DEBUG_CONTEXT)
        .value("OPENGL_PROFILE", MxWindowAttributes::MX_OPENGL_PROFILE)
        .value("CONTEXT_RELEASE_BEHAVIOR", MxWindowAttributes::MX_CONTEXT_RELEASE_BEHAVIOR)
        .value("CONTEXT_NO_ERROR", MxWindowAttributes::MX_CONTEXT_NO_ERROR)
        .value("CONTEXT_CREATION_API", MxWindowAttributes::MX_CONTEXT_CREATION_API)
        .value("SCALE_TO_MONITOR", MxWindowAttributes::MX_SCALE_TO_MONITOR)
        .value("COCOA_RETINA_FRAMEBUFFER", MxWindowAttributes::MX_COCOA_RETINA_FRAMEBUFFER)
        .value("COCOA_FRAME_NAME", MxWindowAttributes::MX_COCOA_FRAME_NAME)
        .value("COCOA_GRAPHICS_SWITCHING", MxWindowAttributes::MX_COCOA_GRAPHICS_SWITCHING)
        .value("X11_CLASS_NAME", MxWindowAttributes::MX_X11_CLASS_NAME)
        .value("X11_INSTANCE_NAME", MxWindowAttributes::MX_X11_INSTANCE_NAME)
        .export_values();

    sim.def_static("window_attrbute", [](MxWindowAttributes attr) -> int {
            SIMULATOR_CHECK();
            return Simulator->app->windowAttribute(attr);
        }
    );

    sim.def_static("set_window_attrbute", [](MxWindowAttributes attr, int val) -> int {
            SIMULATOR_CHECK();
            return Simulator->app->setWindowAttribute(attr, val);
        }
    );
    


    return S_OK;
}

CAPI_FUNC(MxSimulator*) MxSimulator_New(PyObject *_args, PyObject *_kw_args)
{
    return NULL;
}

CAPI_FUNC(MxSimulator*) MxSimulator_Get()
{
    return Simulator;
}


PyObject *not_initialized_error() {
    PyErr_SetString((PyObject*)&MxSimulator_Type, "simulator not initialized");
    Py_RETURN_NONE;
}

CAPI_FUNC(HRESULT) MxSimulator_PollEvents()
{
    SIMULATOR_CHECK();
    return Simulator->app->pollEvents();
}

CAPI_FUNC(HRESULT) MxSimulator_WaitEvents()
{
    SIMULATOR_CHECK();
    return Simulator->app->waitEvents();
}

CAPI_FUNC(HRESULT) MxSimulator_WaitEventsTimeout(double timeout)
{
    SIMULATOR_CHECK();
    return Simulator->app->waitEventsTimeout(timeout);
}

CAPI_FUNC(HRESULT) MxSimulator_PostEmptyEvent()
{
    SIMULATOR_CHECK();
    return Simulator->app->postEmptyEvent();
}

HRESULT MxSimulator_SwapInterval(int si)
{
    SIMULATOR_CHECK();
    return Simulator->app->setSwapInterval(si);
}


int universe_init (const MxUniverseConfig &conf ) {

    Magnum::Vector3 tmp = conf.dim - conf.origin;
    Magnum::Vector3d length{tmp[0], tmp[1], tmp[2]};
    Magnum::Vector3i cells = conf.spaceGridSize;
    
    Magnum::Vector3d spaceGridSize{(float)cells[0],
                                   (float)cells[1],
                                   (float)cells[2]};

    double   cutoff = conf.cutoff;

    int  nr_runners = conf.threads;

    double _origin[3];
    double _dim[3];
    for(int i = 0; i < 3; ++i) {
        _origin[i] = conf.origin[i];
        _dim[i] = conf.dim[i];
    }


    printf("main: initializing the engine... "); fflush(stdout);
    if ( engine_init( &_Engine , _origin , _dim , cells.data() , cutoff , conf.boundaryConditionsPtr ,
            conf.maxTypes , engine_flag_none ) != 0 ) {
        throw std::runtime_error(errs_getstring(0));
    }

    _Engine.dt = conf.dt;
    _Engine.temperature = conf.temp;
    _Engine.integrator = conf.integrator;
    
    _Engine.timers_mask = conf.timers_mask;
    _Engine.timer_output_period = conf.timer_output_period;
    
    if(conf.max_distance >= 0) {
        // max_velocity is in absolute units, convert
        // to scale fraction.
        
        _Engine.particle_max_dist_fraction = conf.max_distance / _Engine.s.h[0];
    }
    
    const char* inte = NULL;

    switch(_Engine.integrator) {
    case EngineIntegrator::FORWARD_EULER:
        inte = "Forward Euler";
        break;
    case EngineIntegrator::RUNGE_KUTTA_4:
        inte = "Ruge-Kutta-4";
        break;
    }

    printf("engine integrator: %s \n", inte);
    printf("engine: n_cells: %i, cell width set to %22.16e.\n", _Engine.s.nr_cells, cutoff);
    printf("engine: cell dimensions = [ %i , %i , %i ].\n", _Engine.s.cdim[0] , _Engine.s.cdim[1] , _Engine.s.cdim[2] );
    printf("engine: cell size = [ %e , %e , %e ].\n" , _Engine.s.h[0] , _Engine.s.h[1] , _Engine.s.h[2] );
    printf("engine: cutoff set to %22.16e.\n", cutoff);
    printf("engine: nr tasks: %i.\n",_Engine.s.nr_tasks);
    printf("engine: nr cell pairs: %i.\n",_Engine.s.nr_pairs);
    
    
    printf("engine: dt: %22.16e.\n",_Engine.dt);
    printf("engine: max distance fraction: %22.16e.\n",_Engine.particle_max_dist_fraction);
    
    // start the engine

    if ( engine_start( &_Engine , nr_runners , nr_runners ) != 0 ) {
        throw std::runtime_error(errs_getstring(0));
    }
    
    fflush(stdout);

    return 0;
}



static std::vector<Vector3> fillCubeRandom(const Vector3 &corner1, const Vector3 &corner2, int nParticles) {
    std::vector<Vector3> result;

    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<float> disx(corner1[0], corner2[0]);
    std::uniform_real_distribution<float> disy(corner1[1], corner2[1]);
    std::uniform_real_distribution<float> disz(corner1[2], corner2[2]);

    for(int i = 0; i < nParticles; ++i) {
        result.push_back(Vector3{disx(gen), disy(gen), disz(gen)});

    }

    return result;
}

CAPI_FUNC(HRESULT) MxSimulator_Run(double et)
{
    SIMULATOR_CHECK();

    std::cout << "simulator run(" << et << ")" << std::endl;

    return Simulator->app->run(et);
}

CAPI_FUNC(HRESULT) MxSimulator_InteractiveRun()
{
    SIMULATOR_CHECK();

    MxUniverse_SetFlag(MX_RUNNING, true);


    std::fprintf(stderr, "checking for ipython \n");
    if (Mx_IsIpython()) {

        if (!MxUniverse_Flag(MxUniverse_Flags::MX_IPYTHON_MSGLOOP)) {
            // ipython message loop, this exits right away
            simulator_interactive_run();
        }

        std::fprintf(stderr, "in ipython, calling interactive \n");

        Simulator->app->show();

        return S_OK;
    }
    else {
        std::fprintf(stderr, "not ipython, returning MxSimulator_Run \n");
        return MxSimulator_Run(-1);
    }
}

static HRESULT simulator_init(py::args args, py::kwargs kwargs) {
    try {
        
        if(Simulator) {
            throw std::domain_error( "Error, Simulator is already initialized" );
        }
        
        MxSimulator *sim = new MxSimulator();
        
        // get the argv,
        py::list argv;
        if(kwargs.contains("argv")) {
            argv = kwargs["argv"];
        }
        else {
            argv = py::module::import("sys").attr("argv");
        }
        
        PyObject *a = argv.ptr();
        
        if(PyList_Size(a) > 0) {
            Universe.name = mx::cast<std::string>(PyList_GetItem(a, 0));
        }
        
  
        
        MxSimulator::Config conf;
        MxSimulator::GLConfig glConf;
        
        if(kwargs.size() > 0 && args.size() == 0) {
            parse_kwargs(kwargs.ptr(), conf);
        }
        else {
            if(args.size() > 0) {
                conf = args[0].cast<MxSimulator::Config>();
            }
            
            if(args.size() > 1) {
                glConf = args[1].cast<MxSimulator::GLConfig>();
            }
        }
        
        // init the engine first
        /* Initialize scene particles */
        universe_init(conf.universeConfig);
        
        if(conf.windowless()) {
            ArgumentsWrapper<MxWindowlessApplication::Arguments> margs(argv);
            
            std::cout << "creating Windowless app" << std::endl;
            
            MxWindowlessApplication *windowlessApp = new MxWindowlessApplication(*margs.pArgs);
            
            if(FAILED(windowlessApp->createContext(conf))) {
                delete windowlessApp;
                
                throw std::domain_error("could not create windowless gl context");
            }
            else {
                sim->app = windowlessApp;
            }
        }
        else {
            ArgumentsWrapper<MxGlfwApplication::Arguments> margs(argv);
            
            std::cout << "creating GLFW app" << std::endl;
            
            MxGlfwApplication *glfwApp = new MxGlfwApplication(*margs.pArgs);
            
            glfwApp->createContext(conf);
            
            sim->app = glfwApp;
        }
        
        std::cout << MX_FUNCTION << std::endl;
        
        Simulator = sim;
        return S_OK;
    }
    catch(const std::exception &e) {
        return C_EXP(e);
    }
}


static void simulator_interactive_run() {
    std::cout << "entering " << MX_FUNCTION << std::endl;
    PYSIMULATOR_CHECK();

    if (MxUniverse_Flag(MxUniverse_Flags::MX_POLLING_MSGLOOP)) {
        return;
    }
    
    // interactive run only works in terminal ipytythn.
    PyObject *ipy = CIPython_Get();
    const char* ipyname = ipy ? ipy->ob_type->tp_name : "NULL";
    std::cerr << "ipy type: " << ipyname << std::endl;
    
    if(ipy && strcmp("TerminalInteractiveShell", ipy->ob_type->tp_name) == 0) {
        
        std::cerr << "calling python interactive loop" << std::endl;

        // Try to import ipython

        /**
         *        """
            Registers the mechanica input hook with the ipython pt_inputhooks
            class.

            The ipython TerminalInteractiveShell.enable_gui('name') method
            looks in the registered input hooks in pt_inputhooks, and if it
            finds one, it activtes that hook.

            To acrtivate the gui mode, call:

            ip = IPython.get_ipython()
            ip.
            """
            import IPython.terminal.pt_inputhooks as pt_inputhooks
            pt_inputhooks.register("mechanica", inputhook)
         *
         */
        
        py::object pt_inputhooks = py::module::import("IPython.terminal.pt_inputhooks");
        py::object reg = pt_inputhooks.attr("register");

        py::cpp_function ih(ipythonInputHook);
        reg("mechanica", ih);

        // import IPython
        // ip = IPython.get_ipython()
        py::object ipython = py::module::import("IPython");
        py::object get_ipython = ipython.attr("get_ipython");
        py::object ip = get_ipython();

        py::object enable_gui = ip.attr("enable_gui");

        enable_gui("mechanica");

        MxUniverse_SetFlag(MxUniverse_Flags::MX_IPYTHON_MSGLOOP, true);
        
        // show the app
        Simulator->app->show();
    }
    else {
        // not in ipython, so run regular run.
        MxSimulator_Run(-1);
        return;
    }
    
    Py_XDECREF(ipy);
    std::cerr << "leaving " << MX_FUNCTION << std::endl;
}

static void ipythonInputHook(py::args args) {
    py::object context = args[0];
    py::object input_is_ready = context.attr("input_is_ready");

    while(!input_is_ready().cast<bool>()) {
        Simulator->app->mainLoopIteration(0.001);
    }
}


CAPI_FUNC(HRESULT) MxSimulator_Show()
{
    SIMULATOR_CHECK();

    std::fprintf(stderr, "checking for ipython \n");
    if (Mx_IsIpython()) {

        if (!MxUniverse_Flag(MxUniverse_Flags::MX_IPYTHON_MSGLOOP)) {
            // ipython message loop, this exits right away
            simulator_interactive_run(); 
        }

        std::fprintf(stderr, "in ipython, calling interactive \n");

        Simulator->app->show();

        return S_OK;
    } 
    else {
        std::fprintf(stderr, "not ipython, returning Simulator->app->show() \n");
        return Simulator->app->show();
    }
}

CAPI_FUNC(HRESULT) MxSimulator_Redraw()
{
    SIMULATOR_CHECK();
    return Simulator->app->redraw();
}

CAPI_FUNC(HRESULT) MxSimulator_InitConfig(const MxSimulator::Config &conf, const MxSimulator::GLConfig &glConf)
{
    if(Simulator) {
        return mx_error(E_FAIL, "simulator already initialized");
    }

    MxSimulator *sim = new MxSimulator();

    // init the engine first
    /* Initialize scene particles */
    universe_init(conf.universeConfig);


    if(conf.windowless()) {

        /*



        MxWindowlessApplication::Configuration windowlessConf;

        MxWindowlessApplication *windowlessApp = new MxWindowlessApplication(*margs.pArgs);

        if(!windowlessApp->tryCreateContext(conf)) {
            delete windowlessApp;

            throw std::domain_error("could not create windowless gl context");
        }
        else {
            sim->app = windowlessApp;
        }
        */
    }
    else {

        std::cout << "creating GLFW app" << std::endl;

        int argc = conf.argc;

        MxGlfwApplication::Arguments args{argc, conf.argv};

        MxGlfwApplication *glfwApp = new MxGlfwApplication(args);

        glfwApp->createContext(conf);

        sim->app = glfwApp;
    }

    std::cout << MX_FUNCTION << std::endl;

    Simulator = sim;

    return S_OK;
}

CAPI_FUNC(HRESULT) MxSimulator_Close()
{
    SIMULATOR_CHECK();
    return Simulator->app->close();
}

CAPI_FUNC(HRESULT) MxSimulator_Destroy()
{
    SIMULATOR_CHECK();
    return Simulator->app->destroy();
}


CAPI_FUNC(bool) Mx_IsIpython() {
    PyObject* ipy = CIPython_Get();
    bool result = false;

    if (ipy && strcmp("TerminalInteractiveShell", ipy->ob_type->tp_name) == 0) {
        result = true;
    }

    Py_XDECREF(ipy);
    return result;
}


/**
 * gets the global simulator object, throws exception if fail.
 */
MxSimulator *MxSimulator::Get() {
    if(Simulator) {
        return Simulator;
    }
    throw std::logic_error("Simulator is not initiazed");
}
