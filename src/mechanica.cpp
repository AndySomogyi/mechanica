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
#include "rendering/MxWindow.h"
#include <rendering/MxGlfwWindow.h>
#include "rendering/MxWindowProxy.h"
#include "rendering/MxWindowHost.h"
#include <rendering/MxUniverseRenderer.h>
#include <MxForce.h>
#include <MxParticleEvent.h>
#include <MxReactivePotential.h>
#include "MxUtil.h"
#include <rendering/NOMStyle.hpp>
#include "MxCluster.hpp"
#include "MxConstraint.hpp"
#include "MxConvert.hpp"
#include "MxParticleList.hpp"
#include <rendering/MxColorMapper.hpp>
#include <rendering/MxKeyEvent.hpp>
#include <MxSecreteUptake.hpp>
#include <DissapativeParticleDynamics.hpp>
#include <MxBoundaryConditions.hpp>
#include <metrics.h>

#include "Vertex.hpp"
#include "Edge.hpp"
#include "Polygon.hpp"
#include "Cell.hpp"
#include "Flux.hpp"
#include "MxBody.hpp"
#include "MxCuboid.hpp"

#include <c_util.h>


#define PY_ARRAY_UNIQUE_SYMBOL MECHANICA_ARRAY_API
#include "numpy/arrayobject.h"


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

        int n = mx::arg<int>("n", 0, args, kwargs, 1);

        uint64_t start = mx::arg<uint64_t>("start", 1, args, kwargs, 2);

        int typenum = NPY_UINT64;

        npy_intp dims[] = {n};


        PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNew(1, dims, typenum);

        uint64_t *data = (uint64_t*)PyArray_DATA(array);

        CMath_FindPrimes(start, n, data);

        return (PyObject*)array;

    }
    catch (const std::exception &e) {
        C_EXP(e); return NULL;
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
    { "init", (PyCFunction)MxSimulator_Init, METH_VARARGS | METH_KEYWORDS, NULL },
    { "pollEvents", (PyCFunction)MxPyUI_PollEvents, METH_NOARGS, NULL },
    { "waitEvents", (PyCFunction)MxPyUI_WaitEvents, METH_VARARGS, NULL },
    { "postEmptyEvent", (PyCFunction)MxPyUI_PostEmptyEvent, METH_NOARGS, NULL },
    { "initializeGraphics", (PyCFunction)MxPyUI_InitializeGraphics, METH_VARARGS, NULL },
    { "createTestWindow", (PyCFunction)MxPyUI_CreateTestWindow, METH_VARARGS, NULL },
    { "testWin", (PyCFunction)PyTestWin, METH_VARARGS, NULL },
    { "destroyTestWindow", (PyCFunction)MxPyUI_DestroyTestWindow, METH_VARARGS, NULL },
    { "on_time", (PyCFunction)MxOnTime, METH_VARARGS | METH_KEYWORDS, NULL },
    { "on_keypress", (PyCFunction)MxKeyEvent_AddDelegate, METH_VARARGS | METH_KEYWORDS, NULL },
    { "invoke_time", (PyCFunction)MxInvokeTime, METH_VARARGS | METH_KEYWORDS, NULL },
    { "random_points", (PyCFunction)MxRandomPoints, METH_VARARGS | METH_KEYWORDS, NULL },
    { "points", (PyCFunction)MxPoints, METH_VARARGS | METH_KEYWORDS, NULL },
    { "bind", (PyCFunction)MxBind, METH_VARARGS | METH_KEYWORDS, NULL },
    { "bind_pairwise", (PyCFunction)bind_pairwise, METH_VARARGS | METH_KEYWORDS, NULL },
    { "bind_sphere", (PyCFunction)bind_sphere, METH_VARARGS | METH_KEYWORDS, NULL },
    { "primes", (PyCFunction)primes, METH_VARARGS | METH_KEYWORDS, NULL },
    { "test", (PyCFunction)MxTest, METH_VARARGS | METH_KEYWORDS, NULL },
    { "virial", (PyCFunction)virial, METH_VARARGS | METH_KEYWORDS, NULL },
    { "flux", (PyCFunction)MxFluxes_Fick, METH_VARARGS | METH_KEYWORDS, NULL },
    { "fick_flux", (PyCFunction)MxFluxes_Fick, METH_VARARGS | METH_KEYWORDS, NULL },
    { "secrete_flux", (PyCFunction)MxFluxes_Secrete, METH_VARARGS | METH_KEYWORDS, NULL },
    { "uptake_flux", (PyCFunction)MxFluxes_Uptake, METH_VARARGS | METH_KEYWORDS, NULL },
    { "produce_flux", (PyCFunction)MxFluxes_Secrete, METH_VARARGS | METH_KEYWORDS, NULL },
    { "consume_flux", (PyCFunction)MxFluxes_Uptake, METH_VARARGS | METH_KEYWORDS, NULL },
    { "reset_species", (PyCFunction)MxUniverse_ResetSpecies, METH_VARARGS | METH_KEYWORDS, NULL },

    { NULL, NULL, 0, NULL }
};



static PyMethodDef version_methods[] = {
    { "cpuinfo", (PyCFunction)MxInstructionSetFeatruesDict, METH_NOARGS, NULL },
    { "compile_flags", (PyCFunction)MxCompileFlagsDict, METH_NOARGS, NULL },
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
        Log(LOG_ERROR) << "could not add version info string";
        return NULL;
    }

    if(PyModule_AddObject(m, "system_name", PyUnicode_FromString(MX_SYSTEM_NAME)) != 0) {
        Log(LOG_ERROR) << "could not add version info string";
        return NULL;
    }

    if(PyModule_AddObject(m, "system_version", PyUnicode_FromString(MX_SYSTEM_VERSION)) != 0) {
        Log(LOG_ERROR) << "could not add version info string";
        return NULL;
    }

    if(PyModule_AddObject(m, "compiler", PyUnicode_FromString(MX_COMPILER_ID)) != 0) {
        Log(LOG_ERROR) << "could not add version info string";
        return NULL;
    }

    if(PyModule_AddObject(m, "compiler_version", PyUnicode_FromString(MX_COMPILER_VERSION)) != 0) {
        Log(LOG_ERROR) << "could not add version info string";
        return NULL;
    }

    std::string datetime = std::string(__DATE__)+ ", " + __TIME__;

    if(PyModule_AddObject(m, "build_date", PyUnicode_FromString(datetime.c_str())) != 0) {
            Log(LOG_ERROR) << "could not add version info string";
            return NULL;
    }

    if(PyModule_AddObject(m, "major", PyLong_FromLong(MX_VERSION_MAJOR)) != 0) {
                Log(LOG_ERROR) << "could not add version info string";
                return NULL;
    }

    if(PyModule_AddObject(m, "minor", PyLong_FromLong(MX_VERSION_MINOR)) != 0) {
                Log(LOG_ERROR) << "could not add version info string";
                return NULL;
    }

    if(PyModule_AddObject(m, "patch", PyLong_FromLong(MX_VERSION_PATCH)) != 0) {
                Log(LOG_ERROR) << "could not add version info string";
                return NULL;
    }

    if(PyModule_AddObject(m, "dev", PyLong_FromLong(MX_VERSION_DEV)) != 0) {
                Log(LOG_ERROR) << "could not add version info string";
                return NULL;
    }

    return m;
}

static PyObject *mechanicaModule = NULL;

CAPI_FUNC(PyObject*) Mx_GetModule() {
    return mechanicaModule;
}


static PyObject * moduleinit(void)
{
    Log(LOG_DEBUG) << ", initializing numpy... " ;

    /* Load all of the `numpy` functionality. */
    import_array();

    PyObject *m;

    // need to initialize the base carbon library first.
    PyObject *carbonModule = PyInit_carbon();

    if(carbonModule == NULL) {
        Log(LOG_FATAL) << "could not initialize carbon: ";
        return NULL;
    }

    m = PyModule_Create(&mechanica_module);
    
    // make a reference to a logger to the carbon logger
    PyObject *logger = PyObject_GetAttrString(carbonModule, "Logger");
    if(logger == NULL) {
        Log(LOG_FATAL) << "could not get Logger from carbon module" ;
    }

    if (m == NULL) {
        Log(LOG_FATAL) << "could not create mechanica module: " ;
        return NULL;
    }
    
    if(PyModule_AddObject(m, "Logger", logger) != 0) {
        Log(LOG_FATAL) << "could not add logger to mechanica module";
    }

    if(PyModule_AddObject(m, "__version__", PyUnicode_FromString(version_str().c_str())) != 0) {
        Log(LOG_FATAL) << "could not add version" ;
        return NULL;
    }

    if(PyModule_AddObject(m, "version", version_create()) != 0) {
        Log(LOG_FATAL) << "error creating version info module";
        return NULL;
    }

    if(PyModule_AddObject(m, "carbon", carbonModule) != 0) {
        Log(LOG_FATAL) << "could not add carbon module ";
        return NULL;
    }

    _MxUtil_init(m);

    // needs to be before other stuff like particles that depend on style.
    _NOMStyle_init(m);
    MxModel_init(m);
    _MxSystem_init(m);
    MxPropagator_init(m);

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
    _MxColormap_Init(m);
    
    _MxKeyEvent_Init(m);

    _MxForces_init(m);

    _MxBond_init(m);
    
    _MxAngle_init(m);
    
    _MxBody_Init(m);
    
    _MxCuboid_Init(m);
    
    _vertex_init(m);
    _edge_init(m);
    _polygon_init(m);
    _cell_init(m);
    
    _MxSecreteUptake_Init(m);
    
    _DPDPotential_Init(m);
    
    _MxBoundaryConditions_Init(m);

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

    Log(LOG_TRACE);

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






