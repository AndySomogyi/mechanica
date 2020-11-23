/*
 * MxConvert.cpp
 *
 *  Created on: Nov 18, 2020
 *      Author: andy
 */

#include "MxConvert.hpp"
#include "MxNumpy.h"
#include <iostream>

namespace mx {
    
//template PyObject* cast<PyObject*>(const Magnum::Vector3 &v);
//template Magnum::Vector3 cast(PyObject *obj);


    
/**
 * convert vector to numpy array
 */

template<>
PyObject* cast(const Magnum::Vector3 &v) {
    // PyArray_SimpleNewFromData(int nd, npy_intp const* dims, int typenum, void* data)
    npy_intp dims = 3;
    float *data = const_cast<float *>(v.data());
    PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNew(1, &dims, NPY_FLOAT);
    
    float *adata = (float*)PyArray_DATA(array);
    
    for(int i = 0; i < 3; ++i) {
        adata[i] = data[i];
    }
    
    return (PyObject*)array;
}
    
template<>
PyObject* cast(const Magnum::Matrix3 &m) {
    // PyArray_SimpleNewFromData(int nd, npy_intp const* dims, int typenum, void* data)
    npy_intp dims[] = {3, 3};
    float *data = const_cast<float *>(m.data());
    PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_FLOAT);
    
    float *adata = (float*)PyArray_DATA(array);
    
    for(int i = 0; i < 9; ++i) {
        adata[i] = data[i];
    }
    
    return (PyObject*)array;
}
    
Magnum::Vector3 vector3_from_list(PyObject *obj) {
    Magnum::Vector3 result = {};
    
    if(PyList_Size(obj) != 3) {
        throw std::domain_error("error, must be length 3 list to convert to vector3");
    }
    
    for(int i = 0; i < 3; ++i) {
        PyObject *item = PyList_GetItem(obj, i);
        if(PyNumber_Check(item)) {
            result[i] = PyFloat_AsDouble(item);
        }
        else {
            throw std::domain_error("error, can not convert list item to number");
        }
    }
    
    return result;
}
    
Magnum::Vector3i vector3i_from_list(PyObject *obj) {
    Magnum::Vector3i result = {};
    
    if(PyList_Size(obj) != 3) {
        throw std::domain_error("error, must be length 3 list to convert to vector3");
    }
    
    for(int i = 0; i < 3; ++i) {
        PyObject *item = PyList_GetItem(obj, i);
        if(PyNumber_Check(item)) {
            result[i] = PyLong_AsLong(item);
        }
        else {
            throw std::domain_error("error, can not convert list item to number");
        }
    }
    
    return result;
}

    
Magnum::Vector3 vector3_from_array(PyObject *obj) {
    Magnum::Vector3 result = {};
    
    npy_intp dims[1] = {3};
    PyArrayObject* tmp = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_FLOAT);
    
    if( PyArray_CopyInto(tmp, (PyArrayObject*)obj)) {
        float *data = (float*)PyArray_GETPTR1(tmp, 0);
        for(int i = 0; i < 3; ++i) {
            result[i] = data[i];
        }
    }
    else {
        Py_DecRef((PyObject*)tmp);
        throw std::domain_error("could not convert array to float array");
        PyErr_Clear();
    }
    
    Py_DecRef((PyObject*)tmp);
    return result;
}
    
Magnum::Vector3i vector3i_from_array(PyObject *obj) {
    Magnum::Vector3i result = {};
    
    npy_intp dims[1] = {3};
    PyArrayObject* tmp = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_INT64);
    
    if( PyArray_CopyInto(tmp, (PyArrayObject*)obj)) {
        int64_t *data = (int64_t*)PyArray_GETPTR1(tmp, 0);
        for(int i = 0; i < 3; ++i) {
            result[i] = data[i];
        }
    }
    else {
        Py_DecRef((PyObject*)tmp);
        throw std::domain_error("could not convert array to int array");
        PyErr_Clear();
    }
    
    Py_DecRef((PyObject*)tmp);
    return result;
}
/**
 * convert vector to numpy array
 */

template<>
Magnum::Vector3 cast(PyObject *obj) {
    if(PyList_Check(obj)) {
        return vector3_from_list(obj);
    }
    
    if(PyArray_Check(obj)) {
        return vector3_from_array(obj);
    }
    throw std::domain_error("can not convert non-list to vector");
}
    
template<>
Magnum::Vector3i cast(PyObject *obj) {
    if(PyList_Check(obj)) {
        return vector3i_from_list(obj);
    }
    
    if(PyArray_Check(obj)) {
        return vector3i_from_array(obj);
    }
    throw std::domain_error("can not convert non-list to vector");
    
}
    
template<>
float cast(PyObject *obj) {
    if(PyNumber_Check(obj)) {
        return PyFloat_AsDouble(obj);
    }
    throw std::domain_error("can not convert to number");
}
    
//template PyObject* cast<PyObject*, const Magnum::Vector3&>(const Magnum::Vector3&);
//template Magnum::Vector3 cast<Magnum::Vector3, PyObject*>(PyObject*);
    
template <>
bool check<bool>(PyObject *o) {
    return PyBool_Check(o);
}
    
template<>
PyObject* cast(const float &f) {
    return PyFloat_FromDouble(f);
}
    
PyObject *arg(const char* name, int index, PyObject *_args, PyObject *_kwargs) {
    PyObject *kwobj = _kwargs ?  PyDict_GetItemString(_kwargs, name) : NULL;
    PyObject *aobj = _args && (PyTuple_Size(_args) > index) ? PyTuple_GetItem(_args, index) : NULL;
    
    if(aobj && kwobj) {
        std::string msg = std::string("Error, argument \"") + name + "\" given both as a keyword and positional";
        throw std::logic_error(msg.c_str());
    }
    
    return aobj ? aobj : kwobj;
}

    
}


