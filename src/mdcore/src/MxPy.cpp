/*
 * MxPy.cpp
 *
 *  Created on: Apr 21, 2020
 *      Author: andy
 */

#include <MxPy.h>
#include <pybind11/pybind11.h>
#include <iostream>

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

