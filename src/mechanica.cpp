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
#include "MxUI.h"
#include "MxTestView.h"
#include "MxApplication.h"
#include "MxSimulator.h"
#include "MxSurfaceSimulator.h"
#include "MxCylinderModel.h"
#include "mdcore_single.h"
#include "MxUniverse.h"

#define PY_ARRAY_UNIQUE_SYMBOL MECHANICA_ARRAY_API
#include "numpy/arrayobject.h"



static PyMethodDef methods[] = {
        { "pollEvents", (PyCFunction)MxPyUI_PollEvents, METH_NOARGS, NULL },
        { "waitEvents", (PyCFunction)MxPyUI_WaitEvents, METH_VARARGS, NULL },
        { "postEmptyEvent", (PyCFunction)MxPyUI_PostEmptyEvent, METH_NOARGS, NULL },
        { "initializeGraphics", (PyCFunction)MxPyUI_InitializeGraphics, METH_VARARGS, NULL },
        { "createTestWindow", (PyCFunction)MxPyUI_CreateTestWindow, METH_VARARGS, NULL },
        { "testWin", (PyCFunction)PyTestWin, METH_VARARGS, NULL },
        { "destroyTestWindow", (PyCFunction)MxPyUI_DestroyTestWindow, METH_VARARGS, NULL },
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

static PyObject *mechanicaModule = NULL;

static PyObject * moduleinit(void)
{
    std::cout << MX_FUNCTION << std::endl;
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




    MxModule_init(m);
    MxModel_init(m);
    MxSystem_init(m);
    MxPropagator_init(m);
    //CObject_init(m);
    //CObject_init(m);

    MxApplication_init(m);
    MxSimulator_init(m);
    MxSurfaceSimulator_init(m);
    MxCylinderModel_init(m);
    MxParticle_init(m);
    MxPotential_init(m);
    MxUniverse_init(m);
    
    mechanicaModule = m;

    return m;
}


PyMODINIT_FUNC PyInit__mechanica(void)
{
    std::cout << MX_FUNCTION << std::endl;
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
    

    if(mechanicaModule == NULL) {
        moduleinit();
    }
    
    return 0;
}






