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

PyMethodDef methods = {
    NULL
};


#if PY_MAJOR_VERSION >= 3
  static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_zope_proxy_proxy", /* m_name */
    module___doc__,      /* m_doc */
    -1,                  /* m_size */
    module_functions,    /* m_methods */
    NULL,                /* m_reload */
    NULL,                /* m_traverse */
    NULL,                /* m_clear */
    NULL,                /* m_free */
  };
#endif

static PyObject * moduleinit(void)
{
    PyObject *m;

#if PY_MAJOR_VERSION >= 3
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule3("mechanica", &methods, "This is a module");
#endif

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

#if PY_MAJOR_VERSION < 3

	/* When Python imports a C module named 'X' it loads the module
	 * then looks for a method named "init"+X and calls it.  Hence
	 * for the module "mechanica" the initialization function is
	 * "initmechanica".
	 */
    PyMODINIT_FUNC initmechanica(void)
    {
    	//import_array();
        moduleinit();
    }
#else
    PyMODINIT_FUNC PyInit_mechanica(void)
    {
        return moduleinit();
    }
#endif







