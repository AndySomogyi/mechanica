/*
 * MxConvert.cpp
 *
 *  Created on: Nov 18, 2020
 *      Author: andy
 */

#include "MxConvert.hpp"
#include <iostream>

// Games with importing numpy and setting up function pointers.
// Only the main Mechanica python init module, mechanica.cpp defines
// MX_IMPORTING_NUMPY_ARRAY and calls import_array()
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL MECHANICA_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "numpy/arrayobject.h"

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
    
//template PyObject* cast<PyObject*, const Magnum::Vector3&>(const Magnum::Vector3&);
//template Magnum::Vector3 cast<Magnum::Vector3, PyObject*>(PyObject*);

    
}




