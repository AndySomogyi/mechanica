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


static void interactiveRun();

static void ipythonInputHook(py::args args);

MxSimulator::Config::Config():
            _title{"Mechanica Application"},
            _size{800, 600},
            _dpiScalingPolicy{DpiScalingPolicy::Default},
            _windowless{false},
            nParticles{100},
            dt{0.01},
            temp{1},
            origin{{0.0, 0.0, 0.0}},
            dim {{10., 10., 10.}} {
    _windowFlags = MxSimulator::WindowFlags::Resizable;
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


/**
 * Create a private 'python' flavored version of the simulator
 * interface here to wrap with pybind11.
 *
 * Don't want to introdude pybind header file into main project and
 * and polute other files.
 */
struct PySimulator : MxSimulator {
    
    PySimulator(py::args args, py::kwargs kwargs) {
        
        if(Simulator) {
            throw std::domain_error( "Error, Simulator is already initialized" );
        }

        bool windowless = false;
        
        // get the argv,
        py::list argv;
        if(kwargs.contains("argv")) {
            argv = kwargs["argv"];
        }
        else {
            argv = py::module::import("sys").attr("argv");
        }
        
        Config conf;
        GLConfig glConf;
        
        if(args.size() > 0) {
            conf = args[0].cast<MxSimulator::Config>();
        }

        if(args.size() > 1) {
            glConf = args[1].cast<MxSimulator::GLConfig>();
        }

        // init the engine first
        /* Initialize scene particles */
        initArgon(conf.origin, conf.dim, conf.nParticles, 0.01, 0.01);

        
        if(conf.windowless()) {
            ArgumentsWrapper<MxWindowlessApplication::Arguments> margs(argv);
            MxWindowlessApplication::Configuration conf;
            
            MxWindowlessApplication *windowlessApp = new MxWindowlessApplication(*margs.pArgs);
            
            if(!windowlessApp->tryCreateContext(conf)) {
                delete windowlessApp;
                
                throw std::domain_error("could not create windowless gl context");
            }
            else {
                this->app = windowlessApp;
            }
        }
        else {
            ArgumentsWrapper<MxGlfwApplication::Arguments> margs(argv);
            
            std::cout << "creating GLFW app" << std::endl;
            
            MxGlfwApplication *glfwApp = new MxGlfwApplication(*margs.pArgs);

            glfwApp->createContext(conf);
            
            this->app = glfwApp;
        }
        
        std::cout << MX_FUNCTION << std::endl;

        Simulator = this;
    }
    
    
    py::handle foo() {
        std::cout << MX_FUNCTION << std::endl;
        PyObject *o = PyLong_FromLong(3);
        
        py::handle h(o);
        
        return h;
    };
    
    ~PySimulator() {
        std::cout << MX_FUNCTION << std::endl;

        assert(Simulator == this);

        Simulator = NULL;
    }

};





static std::string gl_info(const Magnum::Utility::Arguments &args);


static PyObject *not_initialized_error();



MxSimulator* Simulator = NULL;

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




PySimulator *PySimulator_New(py::args args, py::kwargs kwargs) {
    if(!Simulator) {
        Simulator = new PySimulator(args, kwargs);
    }

    return (PySimulator*)Simulator;
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




HRESULT MxSimulator_init(PyObject* m) {

    std::cout << MX_FUNCTION << std::endl;

    py::class_<PySimulator> sim(m, "Simulator");
    sim.def(py::init(&PySimulator_New), py::return_value_policy::reference);
    sim.def_property_readonly("foo", &PySimulator::foo);
    sim.def_static("poll_events", [](){PY_CHECK(MxSimulator_PollEvents());});
    sim.def_static("wait_events", &pysimulator_wait_events);
    sim.def_static("post_empty_event", [](){PY_CHECK(MxSimulator_PostEmptyEvent());});
    sim.def_static("run", [](){PY_CHECK(MxSimulator_Run());});
    sim.def_static("ftest", &ftest);
    sim.def_static("irun", &interactiveRun);


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
            .value("Floating", MxSimulator::WindowFlags::Floating)
            .value("AutoIconify", MxSimulator::WindowFlags::AutoIconify)
            .value("Focused", MxSimulator::WindowFlags::Focused)
            .value("Contextless", MxSimulator::WindowFlags::Contextless)
            .export_values();

    py::class_<MxSimulator::Config> sc(sim, "Config");
    sc.def(py::init());
    sc.def_property("window_title", &MxSimulator::Config::title, &MxSimulator::Config::setTitle);
    sc.def_property("window_size", &MxSimulator::Config::size, &MxSimulator::Config::setSize);
    sc.def_property("dpi_scaling", &MxSimulator::Config::dpiScaling, &MxSimulator::Config::setDpiScaling);
    sc.def_property("window_flags", &MxSimulator::Config::windowFlags, &MxSimulator::Config::setWindowFlags);
    sc.def_property("windowless", &MxSimulator::Config::windowless, &MxSimulator::Config::setWindowless);
    sc.def_readwrite("size", &MxSimulator::Config::nParticles);

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


int initArgon (const Vector3 &origin, const Vector3 &dim,
        int nParticles, double dt, float temp ) {

    double length = dim[0] - origin[0];

    double L[] = { 0.1 * length , 0.1* length , 0.1*  length  };

    double x[3];

    double   cutoff = 0.1 * length;

    struct MxParticle pAr;
    struct MxPotential *pot_ArAr;

    int  k, cid, pid, nr_runners = 8;

    auto pos = fillCubeRandom(origin, dim, nParticles);

    ticks tic, toc;

    tic = getticks();

    double _origin[3];
    double _dim[3];
    for(int i = 0; i < 3; ++i) {
        _origin[i] = origin[i];
        _dim[i] = dim[i];
    }

    // initialize the engine
    printf("main: initializing the engine... ");
    printf("main: requesting origin = [ %f , %f , %f ].\n", _origin[0], _origin[1], _origin[2] );
    printf("main: requesting dimensions = [ %f , %f , %f ].\n", _dim[0], _dim[1], _dim[2] );
    printf("main: requesting cell size = [ %f , %f , %f ].\n", L[0], L[1], L[2] );
    printf("main: requesting cutoff = %22.16e.\n", cutoff);
    fflush(stdout);

    printf("main: initializing the engine... "); fflush(stdout);
    if ( engine_init( &_Engine , _origin , _dim , L , cutoff , space_periodic_full , 2 , engine_flag_none ) != 0 ) {
        printf("main: engine_init failed with engine_err=%i.\n",engine_err);
        errs_dump(stdout);
        return 1;
    }

    _Engine.dt = dt;
    _Engine.temperature = temp;


    printf("main: n_cells: %i, cell width set to %22.16e.\n", _Engine.s.nr_cells, cutoff);

    printf("done.\n"); fflush(stdout);

    // set the interaction cutoff
    printf("main: cell dimensions = [ %i , %i , %i ].\n", _Engine.s.cdim[0] , _Engine.s.cdim[1] , _Engine.s.cdim[2] );
    printf("main: cell size = [ %e , %e , %e ].\n" , _Engine.s.h[0] , _Engine.s.h[1] , _Engine.s.h[2] );
    printf("main: cutoff set to %22.16e.\n", cutoff);
    printf("main: nr tasks: %i.\n",_Engine.s.nr_tasks);

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
    for ( k = 0 ; k < 3 ; k++ ) {
        pAr.x[k] = 0.0;
        pAr.v[k] = 0.0;
        pAr.f[k] = 0.0;
    }

    // create and add the particles
    printf("main: initializing particles... "); fflush(stdout);

    // total velocity squared
    float totV2 = 0;

    for(int i = 0; i < pos.size(); ++i) {
        pAr.id = i;

        pAr.v[0] = ((double)rand()) / RAND_MAX - 0.5;
        pAr.v[1] = ((double)rand()) / RAND_MAX - 0.5;
        pAr.v[2] = ((double)rand()) / RAND_MAX - 0.5;

        totV2 +=   pAr.v[0]*pAr.v[0] + pAr.v[1]*pAr.v[1] + pAr.v[2]*pAr.v[2] ;

        x[0] = pos[i][0];
        x[1] = pos[i][1];
        x[2] = pos[i][2];

        if ( space_addpart( &(_Engine.s) , &pAr , x ) != 0 ) {
            printf("main: space_addpart failed with space_err=%i.\n",space_err);
            errs_dump(stdout);
            return 1;
        }
    }

    float t = (1./ 3.) * _Engine.types[pAr.typeId].mass * totV2 / _Engine.s.nr_parts;
    std::cout << "temperature before scaling: " << t << std::endl;

    float vScale = sqrt((3./_Engine.types[pAr.typeId].mass) * (_Engine.temperature) / (totV2 / _Engine.s.nr_parts));

    // sanity check
    totV2 = 0;

    // scale velocities
    for ( cid = 0 ; cid < _Engine.s.nr_cells ; cid++ ) {
        for ( pid = 0 ; pid < _Engine.s.cells[cid].count ; pid++ ) {
            for ( k = 0 ; k < 3 ; k++ ) {
                _Engine.s.cells[cid].parts[pid].v[k] *= vScale;
                totV2 += _Engine.s.cells[cid].parts[pid].v[k] * _Engine.s.cells[cid].parts[pid].v[k];
            }
        }
    }

    t = (1./ 3.) * _Engine.types[pAr.typeId].mass * totV2 / _Engine.s.nr_parts;
    std::cout << "particle temperature: " << t << std::endl;




    printf("done.\n"); fflush(stdout);
    printf("main: inserted %i particles.\n", _Engine.s.nr_parts);

    // set the time and time-step by hand
    _Engine.time = 0;

    printf("main: dt set to %f fs.\n", _Engine.dt*1000 );

    toc = getticks();

    printf("main: setup took %.3f ms.\n",(double)(toc-tic) * 1000 / CPU_TPS);



    // start the engine

    if ( engine_start( &_Engine , nr_runners , nr_runners ) != 0 ) {
        printf("main: engine_start failed with engine_err=%i.\n",engine_err);
        errs_dump(stdout);
        return 1;
    }


    return 0;
}



void engineStep() {

    //return;

    ticks tic, toc_step, toc_temp;

    double epot, ekin, v2, temp;

    int   k, cid, pid;

    double w;

    // take a step
    tic = getticks();

    //ENGINE_DUMP("pre step: ");

    if ( engine_step( &_Engine ) != 0 ) {
        printf("main: engine_step failed with engine_err=%i.\n",engine_err);
        errs_dump(stdout);
        return ;
    }

    //ENGINE_DUMP("after step: ");

    toc_step = getticks();

    /* Check virtual/local ids. */
    /* for ( cid = 0 ; cid < e.s.nr_cells ; cid++ )
               for ( pid = 0 ; pid < e.s.cells[cid].count ; pid++ )
                   if ( e.s.cells[cid].parts[pid].id != e.s.cells[cid].parts[pid].vid )
                       printf( "main: inconsistent particle id/vid (%i/%i)!\n",
                           e.s.cells[cid].parts[pid].id, e.s.cells[cid].parts[pid].vid ); */

    /* Verify integrity of partlist. */
    /* for ( k = 0 ; k < nr_mols*3 ; k++ )
               if ( e.s.partlist[k]->id != k )
                   printf( "main: inconsistent particle id/partlist (%i/%i)!\n", e.s.partlist[k]->id, k );
           fflush(stdout); */


    // get the total COM-velocities and ekin
    epot = _Engine.s.epot; ekin = 0.0;
#pragma omp parallel for schedule(static,100), private(cid,pid,k,v2), reduction(+:epot,ekin)
    for ( cid = 0 ; cid < _Engine.s.nr_cells ; cid++ ) {
        for ( pid = 0 ; pid < _Engine.s.cells[cid].count ; pid++ ) {
            for ( v2 = 0.0 , k = 0 ; k < 3 ; k++ )
                v2 += _Engine.s.cells[cid].parts[pid].v[k] * _Engine.s.cells[cid].parts[pid].v[k];
            ekin += 0.5 * 39.948 * v2;
        }
    }

    // compute the temperature and scaling
    temp = ekin / ( 1.5 * 6.022045E23 * 1.380662E-26 * _Engine.s.nr_parts );
    w = sqrt( 1.0 + 0.1 * ( _Engine.temperature / temp - 1.0 ) );

    // scale the velocities

    /*
    if ( i < 10000 ) {
#pragma omp parallel for schedule(static,100), private(cid,pid,k), reduction(+:epot,ekin)
        for ( cid = 0 ; cid < _Engine.s.nr_cells ; cid++ ) {
            for ( pid = 0 ; pid < _Engine.s.cells[cid].count ; pid++ ) {
                for ( k = 0 ; k < 3 ; k++ )
                    _Engine.s.cells[cid].parts[pid].v[k] *= w;
            }
        }
    }
     */

    toc_temp = getticks();

    /*

    printf("time:%i, epot:%e, ekin:%e, temp:%e, swaps:%i, stalls: %i %.3f %.3f %.3f ms\n",
            _Engine.time,epot,ekin,temp,_Engine.s.nr_swaps,_Engine.s.nr_stalls,
            (double)(toc_temp-tic) * 1000 / CPU_TPS,
            (double)(toc_step-tic) * 1000 / CPU_TPS,
            (double)(toc_temp-toc_step) * 1000 / CPU_TPS);
    fflush(stdout);
    */

    // print some particle data
    // printf("main: part 13322 is at [ %e , %e , %e ].\n",
    //     e.s.partlist[13322]->x[0], e.s.partlist[13322]->x[1], e.s.partlist[13322]->x[2]);
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
}




static void interactiveRun() {
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
