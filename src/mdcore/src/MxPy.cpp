/*
 * MxPy.cpp
 *
 *  Created on: Apr 21, 2020
 *      Author: andy
 */

#include <MxPy.h>
#include "MxConvert.hpp"
#include <iostream>

bool MxDict_DelItemStringNoErr(PyObject *p, const char *key) {
    PyObject *pkey = mx::cast(std::string(key));
    bool result = false;
    
    if(PyDict_Contains(p, pkey)) {
        PyDict_DelItem(p, pkey);
        result = true;
    }
    
    Py_DECREF(pkey);
    return result;
}

std::ostream& operator<<(std::ostream& os, const PyObject *_obj) {
    if(_obj) {
        PyObject *obj = const_cast<PyObject*>(_obj);
        PyObject *str = PyObject_Str(obj);
        const char* cstr = PyUnicode_AsUTF8(str);
    
        os << obj->ob_type->tp_name << "(" << cstr << ")";
    
        Py_DECREF(str);
    }
    else {
        os << "NULL";
    }
    
    return os;
}

#if PY_MAJOR_VERSION == 3 and PY_MINOR_VERSION < 7

PyObject *PyImport_GetModule(PyObject *name)
{
    PyObject *m;
    PyObject *modules = PyImport_GetModuleDict();
    if (modules == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "unable to get sys.modules");
        return NULL;
    }
    Py_INCREF(modules);
    if (PyDict_CheckExact(modules)) {
        m = PyDict_GetItemWithError(modules, name);  /* borrowed */
        Py_XINCREF(m);
    }
    else {
        m = PyObject_GetItem(modules, name);
        if (m == NULL && PyErr_ExceptionMatches(PyExc_KeyError)) {
            PyErr_Clear();
        }
    }
    Py_DECREF(modules);
    return m;
}
#endif


PyObject *PyImport_ImportString(const char* name) {
    PyObject *s = mx::cast(std::string(name));
    PyObject *mod = PyImport_Import(s);
    Py_DECREF(s);
    return mod;
}




