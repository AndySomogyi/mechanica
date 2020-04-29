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



MxSimulator::Config::Config():
            _title{"Mechanica Application"},
            _size{800, 600},
            _windowFlags{MxSimulator::WindowFlags::Focused},
            _dpiScalingPolicy{DpiScalingPolicy::Default} {}



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
            
            MxGlfwApplication *glfwApp = new MxGlfwApplication(*margs.pArgs, conf);
            
            this->app = glfwApp;
        }



        std::cout << "contains foo: : " << kwargs.contains("foo") << std::endl;
        
        for(auto i : kwargs) {
            
            std::string key  = i.first.cast<std::string>();
            
            std::cout << "key: " << key << std::endl;
            
        }
        
        if(windowless) {

        }
        else  {

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




HRESULT MxSimulator_init(PyObject* m) {

    std::cout << MX_FUNCTION << std::endl;

    py::class_<PySimulator> sim(m, "Simulator");
    sim.def(py::init(&PySimulator_New), py::return_value_policy::reference);
    sim.def_property_readonly("foo", &PySimulator::foo);
    sim.def_static("poll_events", [](){PY_CHECK(MxSimulator_PollEvents());});
    sim.def_static("wait_events", &pysimulator_wait_events);
    sim.def_static("post_empty_event", [](){PY_CHECK(MxSimulator_PostEmptyEvent());});



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

CAPI_FUNC(MxSimulator*) MxSimulator_New(MxSimulator_ConfigurationItem *items)
{
    /*
     *
     *
     *     std::cout << MX_FUNCTION << std::endl;

    if(Simulator) {
        Py_INCREF(Simulator);
        return Simulator;
    }

    int glfw = true;

    if(glfw) {
        int argc = 0;
        char** argv = NULL;
        MxGlfwApplication::Arguments margs(argc, argv);
        MxGlfwApplication *app = new MxGlfwApplication(margs);

        Simulator = (MxSimulator *) type->tp_alloc(type, 0);
        Simulator->applicaiton = app;
        Simulator->kind = MXSIMULATOR_GLFW;
    }
    else {
        int argc = 0;
        char** argv = NULL;
        Platform::WindowlessApplication::Arguments margs(argc, argv);

        app = new MxWindowlessApplication(margs);

        bool result = app->tryCreateContext({});

        if(!result) {
            std::cout << "could not create context..." << std::endl;
            delete app;

            Py_RETURN_NONE;
        }

        Simulator = (MxSimulator *) type->tp_alloc(type, 0);

        Simulator->applicaiton = app;
        Simulator->kind = MXSIMULATOR_WINDOWLESS;
    }

    Py_INCREF(Simulator);
    return (PyObject *) Simulator;
     */
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
