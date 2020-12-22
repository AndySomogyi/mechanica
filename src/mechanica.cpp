/**
 * mechanica.cpp
 *
 * Initialize the mechanica module, python init functions.
 *
 *  Created on: Apr 2, 2017
 *      Author: andy
 */

// only source module that calls import_array()
#define MX_IMPORTING_NUMPY_ARRAY



#include "mechanica_private.h"



#include "MxModule.h"
#include "MxModel.h"
#include "MxSystem.h"
#include "MxPropagator.h"
#include <rendering/MxUI.h>
#include "rendering/MxTestView.h"
#include "rendering/MxApplication.h"
#include "MxSimulator.h"
#include "MxSurfaceSimulator.h"
#include "MxCylinderModel.h"
#include "mdcore_single.h"
#include "MxUniverse.h"
#include "MxUniverseIterators.h"
#include "rendering/MxWindow.h"
#include <rendering/MxGlfwWindow.h>
#include "rendering/MxWindowProxy.h"
#include "rendering/MxWindowHost.h"
#include <rendering/MxUniverseRenderer.h>
#include <MxForce.h>
#include <MxParticleEvent.h>
#include <MxReactivePotential.h>
#include "MxPyTest.h"
#include "MxUtil.h"
#include <rendering/NOMStyle.hpp>
#include "MxCluster.hpp"
#include "MxConstraint.hpp"
#include "MxConvert.hpp"
#include "MxParticleList.hpp"

#include "Vertex.hpp"
#include "Edge.hpp"
#include "Polygon.hpp"
#include "Cell.hpp"


#include <c_util.h>


#define PY_ARRAY_UNIQUE_SYMBOL MECHANICA_ARRAY_API
#include "numpy/arrayobject.h"

#include <pybind11/pybind11.h>

#include <magnum/bootstrap.h>



#include <string>
#include <MxPy.h>

static std::string version_str() {
    #if MX_VERSION_DEV
        std::string dev = "-dev" + std::to_string(MX_VERSION_DEV);
    #else
        std::string dev = "";
    #endif

    std::string version = std::string(MX_VERSION) + dev;
    return version;
}




static PyObject* primes(PyObject *m, PyObject *args, PyObject *kwargs) {

    try {

        int n = arg<int>("n", 0, args, kwargs, 1);

        unsigned long start = arg<unsigned long>("start", 1, args, kwargs, 2);


        int typenum = NPY_UINT64;

        npy_intp dims[] = {n};


        PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNew(1, dims, typenum);

        uint64_t *data = (uint64_t*)PyArray_DATA(array);

        CMath_FindPrimes(start, n, data);

        return (PyObject*)array;

    }
    catch (const std::exception &e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        return NULL;
    }
    catch(pybind11::error_already_set &e){
        e.restore();
        return NULL;
    }

}

static PyObject* MxBind(PyObject *m, PyObject *args, PyObject *kwargs) {
    PyObject *out = NULL;
    HRESULT result = MxUniverse_Bind(args, kwargs, &out);

    if(SUCCEEDED(result)) {
        if(out) {
            return out;
        }
        else {
            Py_RETURN_NONE;
        }
    }
    else {
        return NULL;
    }
}

static PyObject* MxTest(PyObject *m, PyObject *args, PyObject *kwargs) {
    
    Py_RETURN_NONE;
}

/**
 * top level virial tensor method
 */
static PyObject* virial(PyObject *m, PyObject *args, PyObject *kwargs) {
    return NULL;
}

static PyObject *bind_pairwise(PyObject *mod, PyObject *a, PyObject *k) {
    return MxPyUniverse_BindPairwise(a, k);
}


static PyObject *bind_sphere(PyObject *mod, PyObject *a, PyObject *k) {
    return MxUniverse_BindSphere(a, k);
}


static PyMethodDef methods[] = {
        { "pollEvents", (PyCFunction)MxPyUI_PollEvents, METH_NOARGS, NULL },
        { "waitEvents", (PyCFunction)MxPyUI_WaitEvents, METH_VARARGS, NULL },
        { "postEmptyEvent", (PyCFunction)MxPyUI_PostEmptyEvent, METH_NOARGS, NULL },
        { "initializeGraphics", (PyCFunction)MxPyUI_InitializeGraphics, METH_VARARGS, NULL },
        { "createTestWindow", (PyCFunction)MxPyUI_CreateTestWindow, METH_VARARGS, NULL },
        { "testWin", (PyCFunction)PyTestWin, METH_VARARGS, NULL },
        { "destroyTestWindow", (PyCFunction)MxPyUI_DestroyTestWindow, METH_VARARGS, NULL },
        { "on_time", (PyCFunction)MxOnTime, METH_VARARGS | METH_KEYWORDS, NULL },
        { "invoke_time", (PyCFunction)MxInvokeTime, METH_VARARGS | METH_KEYWORDS, NULL },
        { "random_points", (PyCFunction)MxRandomPoints, METH_VARARGS | METH_KEYWORDS, NULL },
        { "points", (PyCFunction)MxPoints, METH_VARARGS | METH_KEYWORDS, NULL },
        { "bind", (PyCFunction)MxBind, METH_VARARGS | METH_KEYWORDS, NULL },
        { "bind_pairwise", (PyCFunction)bind_pairwise, METH_VARARGS | METH_KEYWORDS, NULL },
        { "bind_sphere", (PyCFunction)bind_sphere, METH_VARARGS | METH_KEYWORDS, NULL },
        { "primes", (PyCFunction)primes, METH_VARARGS | METH_KEYWORDS, NULL },
        { "test", (PyCFunction)MxTest, METH_VARARGS | METH_KEYWORDS, NULL },
        { "virial", (PyCFunction)virial, METH_VARARGS | METH_KEYWORDS, NULL },
        { NULL, NULL, 0, NULL }
};



static PyMethodDef version_methods[] = {
    { "cpuinfo", (PyCFunction)MxInstructionSetFeatruesDict, METH_NOARGS, NULL },
    { NULL, NULL, 0, NULL }
};

static struct PyModuleDef mechanica_module = {
        PyModuleDef_HEAD_INIT,
        "_mechanica",   /* name of module */
        NULL, /* module documentation, may be NULL */
        -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
        methods
};

static struct PyModuleDef version_module = {
    PyModuleDef_HEAD_INIT,
    "version",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
               or -1 if the module keeps state in global variables. */
    version_methods
};

static PyObject *version_create() {
    PyObject *m = PyModule_Create(&version_module);

    if(PyModule_AddObject(m, "version", PyUnicode_FromString(version_str().c_str())) != 0) {
        std::cout << "could not add version info string" << std::endl;
        return NULL;
    }

    if(PyModule_AddObject(m, "system_name", PyUnicode_FromString(MX_SYSTEM_NAME)) != 0) {
        std::cout << "could not add version info string" << std::endl;
        return NULL;
    }

    if(PyModule_AddObject(m, "system_version", PyUnicode_FromString(MX_SYSTEM_VERSION)) != 0) {
        std::cout << "could not add version info string" << std::endl;
        return NULL;
    }

    if(PyModule_AddObject(m, "compiler", PyUnicode_FromString(MX_COMPILER_ID)) != 0) {
        std::cout << "could not add version info string" << std::endl;
        return NULL;
    }

    if(PyModule_AddObject(m, "compiler_version", PyUnicode_FromString(MX_COMPILER_VERSION)) != 0) {
        std::cout << "could not add version info string" << std::endl;
        return NULL;
    }

    std::string datetime = std::string(__DATE__)+ ", " + __TIME__;

    if(PyModule_AddObject(m, "build_date", PyUnicode_FromString(datetime.c_str())) != 0) {
            std::cout << "could not add version info string" << std::endl;
            return NULL;
    }

    if(PyModule_AddObject(m, "major", PyLong_FromLong(MX_VERSION_MAJOR)) != 0) {
                std::cout << "could not add version info string" << std::endl;
                return NULL;
    }

    if(PyModule_AddObject(m, "minor", PyLong_FromLong(MX_VERSION_MINOR)) != 0) {
                std::cout << "could not add version info string" << std::endl;
                return NULL;
    }

    if(PyModule_AddObject(m, "patch", PyLong_FromLong(MX_VERSION_PATCH)) != 0) {
                std::cout << "could not add version info string" << std::endl;
                return NULL;
    }

    if(PyModule_AddObject(m, "dev", PyLong_FromLong(MX_VERSION_DEV)) != 0) {
                std::cout << "could not add version info string" << std::endl;
                return NULL;
    }

    return m;
}

static PyObject *mechanicaModule = NULL;

CAPI_FUNC(PyObject*) Mx_GetModule() {
    return mechanicaModule;
}

void test_sequences(PyObject *_m);

static PyObject * moduleinit(void)
{
    std::cout << "Mechanica " << MX_FUNCTION << ", initializing numpy... " << std::endl;
    
    /* Load all of the `numpy` functionality. */
    import_array();
    
    PyObject *m;

    PyObject *carbonModule = PyInit_carbon();


    if(carbonModule == NULL) {
        std::cout << "could not initialize carbon: "  << std::endl;
        return NULL;
    }

    m = PyModule_Create(&mechanica_module);

    if (m == NULL) {
        std::cout << "could not create mechanica module: "  << std::endl;
        return NULL;
    }
    
    if(PyModule_AddObject(m, "__version__", PyUnicode_FromString(version_str().c_str())) != 0) {
        std::cout << "could not add version"  << std::endl;
        return NULL;
    }

    
    if(PyModule_AddObject(m, "version", version_create()) != 0) {
        std::cout << "error creating version info module" << std::endl;
    }

    if(PyModule_AddObject(m, "carbon", carbonModule) != 0) {
        std::cout << "could not add carbon module "  << std::endl;
        return NULL;
    }

    /*

    if (empty_tuple == NULL)
        empty_tuple = PyTuple_New(0);

    ProxyType.tp_free = _PyObject_GC_Del;

    if (PyType_Ready(&ProxyType) < 0)
        return NULL;

    Py_INCREF(&ProxyType);
    PyModule_AddObject(m, "ProxyBase", (PyObject *)&ProxyType);

    if (api_object == NULL) {
        api_object = PyCObject_FromVoidPtr(&wrapper_capi, NULL);
        if (api_object == NULL)
        return NULL;
    }
    Py_INCREF(api_object);
    PyModule_AddObject(m, "_CAPI", api_object);

     */

    pybind11::module rootModule = pybind11::reinterpret_borrow<pybind11::module>(m);

    //pybind11::module rootModule(rootHandle);


    pybind11::module math = rootModule.def_submodule("math", "math module");

    //pybind11::module math = rootModule.def_submodule("math");


    // Create the Magnum math objects and put them in the top-level Mechanica
    // module
    magnum::math(rootModule, math);
    
    // grab the Magnum color3 and color4 pybind types, and add some new constructors
    // to them, to make colors based on name.

    pybind11::object o = (pybind11::object)rootModule.attr("Color3");
    
    pybind11::class_<Magnum::Color3> color3(o);
    
    color3.def(pybind11::init([](std::string arg) -> Magnum::Color3 {
        return Color3_Parse(arg);
    }));
    
    
    pybind11::class_<Magnum::Color4> color4((pybind11::object)rootModule.attr("Color4"));

    color4.def(pybind11::init([](std::string arg) -> Magnum::Color4 {
        return Magnum::Color4(Color3_Parse(arg));
    }));


    _MxUtil_init(m);
    
    // needs to be before other stuff like particles that depend on style. 
    _NOMStyle_init(m);
    MxModule_init(m);
    MxModel_init(m);
    MxSystem_init(m);
    MxPropagator_init(m);
    //CObject_init(m);
    //CObject_init(m);


    _MxSimulator_init(m);
    MxSurfaceSimulator_init(m);
    MxCylinderModel_init(m);
    _MxParticle_init(m);
    _MxParticleList_init(m);
    _MxCluster_init(m);
    _MxPotential_init(m);
    _MxReactivePotential_init(m);
    _MxUniverse_init(m);
    MxWindow_init(m);
    MxGlfwWindow_init(m);
    MxWindowProxy_init(m);
    MxWindowHost_init(m);
    MyUniverseRenderer_Init(m);
    
    

    MxPyTest_init(m);

    test_sequences(m);
    _MxUniverseIterators_init(m);

    _MxForces_init(m);

    _MxBond_init(m);
    
    _MxAngle_init(m);
    

    _vertex_init(m);
    _edge_init(m);
    _polygon_init(m);
    _cell_init(m);

    mechanicaModule = m;

    return m;
}


CAPI_FUNC(PyObject*) PyInit__mechanica(void)
{
    return moduleinit();
}


/**
 * Initialize the entire runtime.
 */
CAPI_FUNC(HRESULT) Mx_Initialize(int args) {

    std::cout << MX_FUNCTION << std::endl;

    HRESULT result = E_FAIL;

    if(!Py_IsInitialized()) {
        Py_Initialize();
    }
    
    // GL symbols are globals in each shared library address space,
    // if the app already initialized gl, we need to get the symbols here
    if(Magnum::GL::Context::hasCurrent() && !glCreateProgram) {
        flextGLInit(Magnum::GL::Context::current());
    }


    if(mechanicaModule == NULL) {
        moduleinit();
    }
    

    return 0;
}






