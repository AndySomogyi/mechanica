/*
 * MxSimulator.cpp
 *
 *  Created on: Feb 1, 2017
 *      Author: andy
 */

#include <MxSimulator.h>
#include <MxUI.h>
#include <MxTestView.h>

#include <Magnum/GL/Context.h>

#if defined(MX_APPLE)
    #include "Magnum/Platform/WindowlessCglApplication.h"
#elif defined(MX_LINUX)
    #include "Magnum/Platform/WindowlessEglApplication.h"
#elif defined(MX_WINDOWS)
    #include "Magnum/Platform/WindowlessWglApplication.h"
#else
    #error no windowless application available on this platform
#endif


#include "Magnum/Platform/GlfwApplication.h"


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




static PyTypeObject SimulatorType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    .tp_name = "mechanica.Simulator",
    .tp_basicsize = sizeof(MxSimulator),
    .tp_itemsize = 0,
    .tp_dealloc = 0,
    .tp_print = 0, 
    .tp_getattr = 0, 
    .tp_setattr = 0, 
    .tp_as_async = 0, 
    .tp_repr = 0, 
    .tp_as_number = 0, 
    .tp_as_sequence = 0, 
    .tp_as_mapping = 0, 
    .tp_hash = 0, 
    .tp_call = 0, 
    .tp_str = 0, 
    .tp_getattro = 0, 
    .tp_setattro = 0, 
    .tp_as_buffer = 0, 
    .tp_flags = Py_TPFLAGS_DEFAULT, 
    .tp_doc = 0, 
    .tp_traverse = 0, 
    .tp_clear = 0, 
    .tp_richcompare = 0, 
    .tp_weaklistoffset = 0, 
    .tp_iter = 0, 
    .tp_iternext = 0, 
    .tp_methods = methods, 
    .tp_members = 0, 
    .tp_getset = 0, 
    .tp_base = 0, 
    .tp_dict = 0, 
    .tp_descr_get = 0, 
    .tp_descr_set = 0, 
    .tp_dictoffset = 0, 
    .tp_init = (initproc)0,
    .tp_alloc = 0, 
    .tp_new = PyType_GenericNew,
    .tp_free = 0,  
    .tp_is_gc = 0, 
    .tp_bases = 0, 
    .tp_mro = 0, 
    .tp_cache = 0, 
    .tp_subclasses = 0, 
    .tp_weaklist = 0, 
    .tp_del = 0, 
    .tp_version_tag = 0, 
    .tp_finalize = 0, 
};

/*
  PyVarObject_HEAD_INIT(NULL, 0)
  tp_name : "mechanica.Simulator",
  tp_basicsize : sizeof(MxSimulator),
  tp_itemsize : 0,
  tp_flags : Py_TPFLAGS_DEFAULT,
  tp_doc : "Custom objects",
  tp_methods : methods,
  tp_init : init,
  tp_new : PyType_GenericNew,
*/


PyTypeObject *MxSimuator_Type = &SimulatorType;



HRESULT MxSimulator_init(PyObject* m) {

    std::cout << MX_FUNCTION << std::endl;


    if (PyType_Ready((PyTypeObject *)MxSimuator_Type) < 0)
        return E_FAIL;



    Py_INCREF(MxSimuator_Type);
    PyModule_AddObject(m, "Simulator", (PyObject *) MxSimuator_Type);

    return 0;
}



