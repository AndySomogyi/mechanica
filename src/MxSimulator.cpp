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

#include <rendering/MxGlfwApplication.h>
#include <rendering/MxWindowlessApplication.h>
#include <map>
#include <sstream>
#include <MxUniverse.h>

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


MxSimulator::Config::Config():
            _title{"Mechanica Application"},
            _size{800, 600},
            _dpiScalingPolicy{DpiScalingPolicy::Default},
            _windowless{false},
            threads{4},
            queues{4} {
    _windowFlags = MxSimulator::WindowFlags::Resizable | MxSimulator::WindowFlags::Focused;
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


static void parse_kwargs(const py::kwargs &kwargs, MxSimulator::Config &conf) {
    if(kwargs.contains("example")) {
        py::object example = kwargs["example"];
        if(py::isinstance<py::none>(example)) {
            conf.example = "";
        }
        conf.example = py::cast<std::string>(kwargs["example"]);
    }
    
    if(kwargs.contains("dim")) {
        conf.universeConfig.dim = py::cast<Vector3>(kwargs["dim"]);
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
        std::cout << MX_FUNCTION << std::endl;
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

static PyObject *Noddy_name(MxSimulator* self)
{
    return PyUnicode_FromFormat("%s %s", "foo", "bar");
}







static PyObject *simulator_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    return NULL;
}





#if 0
PyTypeObject THPLegacyVariableType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "torch._C._LegacyVariableBase",        /* tp_name */
    0,                                     /* tp_basicsize */
    0,                                     /* tp_itemsize */
    0,                                     /* tp_dealloc */
    0,                                     /* tp_print */
    0,                                     /* tp_getattr */
    0,                                     /* tp_setattr */
    0,                                     /* tp_reserved */
    0,                                     /* tp_repr */
    0,                                     /* tp_as_number */
    0,                                     /* tp_as_sequence */
    0,                                     /* tp_as_mapping */
    0,                                     /* tp_hash  */
    0,                                     /* tp_call */
    0,                                     /* tp_str */
    0,                                     /* tp_getattro */
    0,                                     /* tp_setattro */
    0,                                     /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    nullptr,                               /* tp_doc */
    0,                                     /* tp_traverse */
    0,                                     /* tp_clear */
    0,                                     /* tp_richcompare */
    0,                                     /* tp_weaklistoffset */
    0,                                     /* tp_iter */
    0,                                     /* tp_iternext */
    0,                                     /* tp_methods */
    0,                                     /* tp_members */
    0,                                     /* tp_getset */
    0,                                     /* tp_base */
    0,                                     /* tp_dict */
    0,                                     /* tp_descr_get */
    0,                                     /* tp_descr_set */
    0,                                     /* tp_dictoffset */
    0,                                     /* tp_init */
    0,                                     /* tp_alloc */
    0                      /* tp_new */
};
#endif


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
    sim.def_static("run", [](){PY_CHECK(MxSimulator_Run());});
    sim.def_static("ftest", &ftest);
    sim.def_static("irun", &simulator_interactive_run);
    sim.def_static("show", [] () { PY_CHECK(MxSimulator_Show()); });


    sim.def_property_readonly_static("renderer", [](py::object) -> py::handle {
            PYSIMULATOR_CHECK();
            return py::handle(Simulator->app->getRenderer());
        }
    );

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
    
    sc.def_property("boundary_conditions", [](const MxSimulator::Config &conf) { return conf.universeConfig.boundaryConditions; },
                    [](MxSimulator::Config &conf, Magnum::Vector3ui& vec) {conf.universeConfig.boundaryConditions = vec;});
    
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
    



    py::class_<MxSimulator::GLConfig> gc(sim, "GLConfig");
    gc.def(py::init());
    gc.def_property("color_buffer_size", &MxSimulator::GLConfig::colorBufferSize, &MxSimulator::GLConfig::setColorBufferSize);
    gc.def_property("depth_buffer_size", &MxSimulator::GLConfig::depthBufferSize, &MxSimulator::GLConfig::setDepthBufferSize);
    gc.def_property("stencil_buffer_size", &MxSimulator::GLConfig::stencilBufferSize, &MxSimulator::GLConfig::setStencilBufferSize);
    gc.def_property("sample_count", &MxSimulator::GLConfig::sampleCount, &MxSimulator::GLConfig::setSampleCount);
    gc.def_property("srgb_capable", &MxSimulator::GLConfig::isSrgbCapable, &MxSimulator::GLConfig::setSrgbCapable);



    foo(m);




    return 0;
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
    Magnum::Vector3d spaceGridSize{conf.spaceGridSize[0], conf.spaceGridSize[2], conf.spaceGridSize[2]};

    Magnum::Vector3d L = length / spaceGridSize;

    double x[3];

    double   cutoff = 0.1 * length.sum() / 3.;

    int  k, cid, pid, nr_runners = conf.threads;

    ticks tic, toc;

    tic = getticks();

    double _origin[3];
    double _dim[3];
    for(int i = 0; i < 3; ++i) {
        _origin[i] = conf.origin[i];
        _dim[i] = conf.dim[i];
    }

    // initialize the engine
    printf("main: initializing the engine... ");
    printf("main: requesting origin = [ %f , %f , %f ].\n", _origin[0], _origin[1], _origin[2] );
    printf("main: requesting dimensions = [ %f , %f , %f ].\n", _dim[0], _dim[1], _dim[2] );
    printf("main: requesting cell size = [ %f , %f , %f ].\n", L[0], L[1], L[2] );
    printf("main: requesting cutoff = %22.16e.\n", cutoff);
    fflush(stdout);

    printf("main: initializing the engine... "); fflush(stdout);
    if ( engine_init( &_Engine , _origin , _dim , L.data() , cutoff , space_periodic_full ,
            conf.maxTypes , engine_flag_none ) != 0 ) {
        printf("main: engine_init failed with engine_err=%i.\n",engine_err);
        errs_dump(stdout);
        return 1;
    }

    _Engine.dt = conf.dt;
    _Engine.temperature = conf.temp;

    printf("main: n_cells: %i, cell width set to %22.16e.\n", _Engine.s.nr_cells, cutoff);

    printf("done.\n"); fflush(stdout);

    // set the interaction cutoff
    printf("main: cell dimensions = [ %i , %i , %i ].\n", _Engine.s.cdim[0] , _Engine.s.cdim[1] , _Engine.s.cdim[2] );
    printf("main: cell size = [ %e , %e , %e ].\n" , _Engine.s.h[0] , _Engine.s.h[1] , _Engine.s.h[2] );
    printf("main: cutoff set to %22.16e.\n", cutoff);
    printf("main: nr tasks: %i.\n",_Engine.s.nr_tasks);

    // start the engine

    if ( engine_start( &_Engine , nr_runners , nr_runners ) != 0 ) {
        printf("main: engine_start failed with engine_err=%i.\n",engine_err);
        errs_dump(stdout);
        return 1;
    }

    return 0;
}



static std::vector<Vector3> fillCubeRandom(const Vector3 &corner1, const Vector3 &corner2, int nParticles) {
    std::vector<Vector3> result;

    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> disx(corner1[0], corner2[0]);
    std::uniform_real_distribution<> disy(corner1[1], corner2[1]);
    std::uniform_real_distribution<> disz(corner1[2], corner2[2]);

    for(int i = 0; i < nParticles; ++i) {
        result.push_back(Vector3{disx(gen), disy(gen), disz(gen)});

    }

    return result;
}

CAPI_FUNC(HRESULT) MxSimulator_Run()
{
    SIMULATOR_CHECK();
    return Simulator->app->run();
}

CAPI_FUNC(HRESULT) MxSimulator_InteractiveRun()
{
    SIMULATOR_CHECK();
    try {

    }
    catch(std::exception  &err) {

    }
    catch (pybind11::error_already_set &err) {

    }
    catch(...) {

    }
    return E_FAIL;
}

static HRESULT simulator_init(py::args args, py::kwargs kwargs) {

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

    MxSimulator::Config conf;
    MxSimulator::GLConfig glConf;

    if(kwargs.size() > 0 && args.size() == 0) {
        parse_kwargs(kwargs, conf);
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

    if(conf.example.compare("argon")==0) {
        example_argon(conf.universeConfig);
    }

    if(conf.windowless()) {
        ArgumentsWrapper<MxWindowlessApplication::Arguments> margs(argv);
        MxWindowlessApplication::Configuration conf;

        MxWindowlessApplication *windowlessApp = new MxWindowlessApplication(*margs.pArgs);

        if(!windowlessApp->tryCreateContext(conf)) {
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




static void simulator_interactive_run() {
    std::cout << "entering " << MX_FUNCTION << std::endl;
    PYSIMULATOR_CHECK();
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

    std::cout << "leaving " << MX_FUNCTION << std::endl;
}

static void ipythonInputHook(py::args args) {
    py::object context = args[0];
    py::object input_is_ready = context.attr("input_is_ready");

    while(!input_is_ready().cast<bool>()) {
        Simulator->app->mainLoopIteration(0.001);
    }
}

HRESULT example_argon(const MxUniverseConfig &conf) {
    
    std::cout << "loading argon example" << std::endl;

    double length = conf.dim[0] - conf.origin[0];

    double L[] = { 0.1 * length , 0.1* length , 0.1*  length  };

    double x[3];

    double   cutoff = 0.1 * length;

    struct MxParticle pAr;
    struct MxPotential *pot_ArAr;

    auto pos = fillCubeRandom(conf.origin, conf.dim, conf.nParticles);

    /* mix-up the pair list just for kicks
    printf("main: shuffling the interaction pairs... "); fflush(stdout);
    srand(6178);
    for ( i = 0 ; i < e.s.nr_pairs ; i++ ) {
        j = rand() % e.s.nr_pairs;
        if ( i != j ) {
            cp = e.s.pairs[i];
            e.s.pairs[i] = e.s.pairs[j];
            e.s.pairs[j] = cp;
            }
        }
    printf("done.\n"); fflush(stdout); */


    // initialize the Ar-Ar potential
    if ( ( pot_ArAr = potential_create_LJ126( 0.275 , cutoff, 9.5075e-06 , 6.1545e-03 , 1.0e-3 ) ) == NULL ) {
        printf("main: potential_create_LJ126 failed with potential_err=%i.\n",potential_err);
        errs_dump(stdout);
        return 1;
    }
    printf("main: constructed ArAr-potential with %i intervals.\n",pot_ArAr->n); fflush(stdout);


    /* register the particle types. */
    if ( ( pAr.typeId = engine_addtype( &_Engine , 39.948 , 0.0 , "Ar" , "Ar" ) ) < 0 ) {
        printf("main: call to engine_addtype failed.\n");
        errs_dump(stdout);
        return 1;
    }

    // register these potentials.
    if ( engine_addpot( &_Engine , pot_ArAr , pAr.typeId , pAr.typeId ) < 0 ){
        printf("main: call to engine_addpot failed.\n");
        errs_dump(stdout);
        return 1;
    }

    // set fields for all particles
    srand(6178);

    pAr.flags = PARTICLE_FLAG_NONE;
    for (int k = 0 ; k < 3 ; k++ ) {
        pAr.x[k] = 0.0;
        pAr.v[k] = 0.0;
        pAr.f[k] = 0.0;
    }

    // create and add the particles
    printf("main: initializing particles... "); fflush(stdout);

    // total velocity squared
    float totV2 = 0;

    float vscale = 10.0;

    for(int i = 0; i < pos.size(); ++i) {
        pAr.id = i;

        pAr.v[0] = vscale * (((double)rand()) / RAND_MAX - 0.5);
        pAr.v[1] = vscale * (((double)rand()) / RAND_MAX - 0.5);
        pAr.v[2] = vscale * (((double)rand()) / RAND_MAX - 0.5);

        totV2 +=   pAr.v[0]*pAr.v[0] + pAr.v[1]*pAr.v[1] + pAr.v[2]*pAr.v[2] ;

        x[0] = pos[i][0];
        x[1] = pos[i][1];
        x[2] = pos[i][2];

        if (engine_addpart( &_Engine , &pAr , x, NULL ) != 0 ) {
            printf("main: space_addpart failed with space_err=%i.\n",space_err);
            errs_dump(stdout);
            return 1;
        }
    }



    printf("done.\n"); fflush(stdout);
    printf("main: inserted %i particles.\n", _Engine.s.nr_parts);

    // set the time and time-step by hand
    _Engine.time = 0;

    printf("main: dt set to %f fs.\n", _Engine.dt*1000 );

    return S_OK;

}

CAPI_FUNC(HRESULT) MxSimulator_Show()
{
    SIMULATOR_CHECK();

    if(Simulator->flags & MxSimulator::Flags::Running) {
        // TODO: add something to application to show window
        return S_OK;
    }

    MxUniverse_SetFlag(MxUniverse_Flags::MXU_RUNNING, false);

    simulator_interactive_run();

    return S_OK;
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

    if(conf.example.compare("argon")==0) {
        example_argon(conf.universeConfig);
    }

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
