/*
 * MxPyTest.cpp
 *
 *  Created on: Apr 16, 2020
 *      Author: andy
 */

#include <MxPyTest.h>

#include <iostream>

#include <pybind11/pybind11.h>



// test structure definition
struct Foo {
    int x; int y; int z;
};

// call this function once, when the main library is initialized from Python .
static void Foo_init(PyObject *module) {

    pybind11::class_<Foo> foo(module, "Foo");

    foo.def(pybind11::init<>());
    foo.def_readwrite("x", &Foo::x);
    foo.def_readwrite("y", &Foo::y);
    foo.def_readwrite("z", &Foo::z);
}


// test function, declared as a stand alone function
static PyObject *test(PyObject* self, PyObject* args)  {

    Foo *f = new Foo();
    
    pybind11::object o = pybind11::cast(f);
    
    o.inc_ref();
    
    PyObject *p = o.ptr();
    
    return p;
}



static PyMethodDef pytest_methods[] = {
    {"test", (PyCFunction)test, METH_VARARGS,  "make a test image" },
    {NULL}  /* Sentinel */
};





PyTypeObject MxPyTest_Type = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    .tp_name = "Test",
    .tp_basicsize = sizeof(MxPyTest),
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
    .tp_methods = pytest_methods,
    .tp_members = 0,
    .tp_getset = 0,
    .tp_base = 0,
    .tp_dict = 0,
    .tp_descr_get = 0,
    .tp_descr_set = 0,
    .tp_dictoffset = 0,
    .tp_init = 0,
    .tp_alloc = PyType_GenericAlloc,
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

HRESULT MxPyTest_init(PyObject *m){

    std::cout << MX_FUNCTION << std::endl;


    if (PyType_Ready((PyTypeObject *)&MxPyTest_Type) < 0)
        return E_FAIL;



    Py_INCREF(&MxPyTest_Type);
    PyModule_AddObject(m, "Test", (PyObject *) &MxPyTest_Type);

    Foo_init(m);




    return 0;
}
