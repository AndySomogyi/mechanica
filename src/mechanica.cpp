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


static PyObject * moduleinit(void)
{
    std::cout << __PRETTY_FUNCTION__ << std::endl;
    PyObject *m;


    m = PyModule_Create(&mechanica_module);


    if (m == NULL)
        return NULL;

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

    MxObject_init(m);
    MxType_init(m);
    MxSymbol_init(m);
    MxModule_init(m);
    MxModel_init(m);
    MxSystem_init(m);
    MxPropagator_init(m);
    //MxObject_init(m);
    //MxObject_init(m);

    return m;
}


PyMODINIT_FUNC PyInit__mechanica(void)
{
    return moduleinit();
}








