/*
 * MxPy.cpp
 *
 *  Created on: Apr 21, 2020
 *      Author: andy
 */

#include <MxPy.h>
#include <pybind11/pybind11.h>
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

template<typename T>
PyObject *PyBind_Getter(PyObject *obj, bool byReference, size_t offset) {

    char* p = ((char*)obj) + offset;
    T *pMember = (T*)p;

    if(byReference) {

        pybind11::handle h = pybind11::cast(pMember).release();

        PyObject *result = h.ptr();

        std::cout << "result: " << result << std::endl;
        std::cout << "result.refcnt: " << result->ob_refcnt << std::endl;
        std::cout << "result.type: " << result->ob_type->tp_name << std::endl;

        return result;
    }
    else {

        pybind11::handle h = pybind11::cast(*pMember).release();

        PyObject *result = h.ptr();

        std::cout << "result: " << result << std::endl;
        std::cout << "result.refcnt: " << result->ob_refcnt << std::endl;
        std::cout << "result.type: " << result->ob_type->tp_name << std::endl;

        return result;

    }
}

template<typename T>
int PyBind_Setter(PyObject *obj, PyObject *arg, bool byReference, size_t offset) {

    char* p = ((char*)obj) + offset;
    T *pMember = (T*)p;

    pybind11::handle h(arg);

    T x = h.cast<T>(); // may throw cast_error

    *pMember = x;

    return 0;
}


PyObject *_MxGetSetDef_Getter(PyObject *obj, MxGetSetDefInfo info) {

    PyObject *result = NULL;

    switch (info.kind) {
        case MxGetSetDef_Int:
            result = PyBind_Getter<int>(obj, false, info.offset);
            break;
        case MxGetSetDef_Float:
            result = PyBind_Getter<float>(obj, false, info.offset);
            break;
        case MxGetSetDef_Double:
            result = PyBind_Getter<double>(obj, false, info.offset);
            break;
        case MxGetSetDef_Vector3f:
            result = PyBind_Getter<Magnum::Vector3>(obj, true, info.offset);
            break;
        default:
            PyErr_SetString(obj, "invalid data type");
            break;
    }
    return result;
}

int _MxGetSetDef_Setter(PyObject *obj, PyObject *arg, MxGetSetDefInfo info) {
    int result = -1;

    switch (info.kind) {
        case MxGetSetDef_Int:
            result = PyBind_Setter<int>(obj, arg, false, info.offset);
            break;
        case MxGetSetDef_Float:
            result = PyBind_Setter<float>(obj, arg, false, info.offset);
            break;
        case MxGetSetDef_Double:
            result = PyBind_Setter<double>(obj, arg, false, info.offset);
            break;
        case MxGetSetDef_Vector3f:
            result = PyBind_Setter<Magnum::Vector3>(obj, arg, true, info.offset);
            break;
        default:
            PyErr_SetString(obj, "invalid data type");
            break;
    }
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




