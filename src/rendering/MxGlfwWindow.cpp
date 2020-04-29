/*
 * MxWindowNative.cpp
 *
 *  Created on: Apr 13, 2020
 *      Author: andy
 */

#include <rendering/MxGlfwWindow.h>

#include <Python.h>
#include <MxPy.h>
#include <iostream>



/*

PyGetSetDef particle_getsets[] = {
        //create_vector4_getset(),
    MakeAttibute("position", "doc", &MxParticle::position),
    MakeAttibute("velocity", "doc", &MxParticle::velocity),
    MakeAttibute("force", "doc", &MxParticle::force),
    MakeAttibute("q", "doc", &MxParticle::q),
    {NULL}
};

*/

static PyObject*
mytype_get_float_field(MxGlfwWindow* self, void* closure)
{
    return PyFloat_FromDouble(self->f);
}

static int
mytype_set_float_field(MxGlfwWindow* self, PyObject* value, void* closure)
{
    float f = PyFloat_AsDouble(value);
    if (PyErr_Occurred()) {
        return -1;
    }

    if (f < 0) {
        PyErr_SetString(PyExc_ValueError, "float_field must be positive");
    }

    self->f = f;
    return 0;
}




template <class Class, class Result, Result Class::*Member>
struct MyStruct {
    float f;
};

template<typename Klass, typename VarType, VarType Klass::*pm>
void Test(const char* name, const char* doc) {
    std::cout << "foo";
}

void testtt() {
    
    MyStruct<MxGlfwWindow, float, &MxGlfwWindow::f> variable;
    
    Test<MxGlfwWindow, float, &MxGlfwWindow::f>("", "");
    
    PyGetSetDef gs = MakeAttibuteGetSet<MxGlfwWindow, float, &MxGlfwWindow::f>("", "");
    
    
    MxGlfwWindow win;
    
    PyObject *obj = gs.get(&win, NULL);
    
    std::cout << "type: " << obj->ob_type->tp_name << std::endl;
    
    PyObject *n = PyFloat_FromDouble(987);
    
    gs.set(&win, n, NULL);
}

PyGetSetDef glfwwindow_getsets[] = {
    MakeAttibuteGetSet<MxGlfwWindow, float, &MxGlfwWindow::f>("f", "foo doc"),
    {NULL}
};

static int glfwwindow_init(MxGlfwWindow *self, PyObject *args, PyObject *kwds)
{
    
    std::cout << MX_FUNCTION << std::endl;
    
    self->f = 123456789.0;
    
    testtt();

    
    return 0;
}

PyTypeObject MxGlfwWindow_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name =           "GlfwWindow",
    .tp_basicsize =      sizeof(MxGlfwWindow),
    .tp_itemsize =       0,
    .tp_dealloc =        0,
    .tp_print =          0,
    .tp_getattr =        0,
    .tp_setattr =        0,
    .tp_as_async =       0,
    .tp_repr =           0,
    .tp_as_number =      0,
    .tp_as_sequence =    0,
    .tp_as_mapping =     0,
    .tp_hash =           0,
    .tp_call =           0,
    .tp_str =            0,
    .tp_getattro =       0,
    .tp_setattro =       0,
    .tp_as_buffer =      0,
    .tp_flags =          Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc =            "Custom objects",
    .tp_traverse =       0,
    .tp_clear =          0,
    .tp_richcompare =    0,
    .tp_weaklistoffset = 0,
    .tp_iter =           0,
    .tp_iternext =       0,
    .tp_methods =        0,
    .tp_members =        0,
    .tp_getset =         glfwwindow_getsets,
    .tp_base =           0,
    .tp_dict =           0,
    .tp_descr_get =      (descrgetfunc)0,
    .tp_descr_set =      0,
    .tp_dictoffset =     0,
    .tp_init =           (initproc)glfwwindow_init,
    .tp_alloc =          PyType_GenericAlloc,
    .tp_new =            PyType_GenericNew,
    .tp_free =           0,
    .tp_is_gc =          0,
    .tp_bases =          0,
    .tp_mro =            0,
    .tp_cache =          0,
    .tp_subclasses =     0,
    .tp_weaklist =       0,
    .tp_del =            0,
    .tp_version_tag =    0,
    .tp_finalize =       0,
};


HRESULT MxGlfwWindow_init(PyObject *m)
{
    if (PyType_Ready(&MxGlfwWindow_Type)) {
        return -1;
    }
    

    
    Py_INCREF(&MxGlfwWindow_Type);
    if (PyModule_AddObject(m, "Window", (PyObject*)&MxGlfwWindow_Type)) {
        Py_DECREF(&MxGlfwWindow_Type);
        return -1;
    }
    return 0;
}
