/*
 * MxConvert.hpp
 *
 *  Created on: Nov 18, 2020
 *      Author: andy
 */

#ifndef SRC_MDCORE_INCLUDE_MXCONVERT_HPP_
#define SRC_MDCORE_INCLUDE_MXCONVERT_HPP_

#include <platform.h>
#include <Magnum/Magnum.h>
#include <Magnum/Math/Vector3.h>
#include <Magnum/Math/Matrix3.h>


namespace mx {

/**
 * convert from c++ to python type
 */
template <typename T>
PyObject *cast(const T& x);
    
/**
 * convert from c++ to python type
 */
//template <typename T>
//PyObject *cast(T x);
    
/**
 * convert from python to c++ type
 */
template <typename T>
T cast(PyObject *o);
    
template<>
PyObject* cast(const Magnum::Vector3 &v);
    
template<>
PyObject* cast(const Magnum::Matrix3 &m);
    
template<>
Magnum::Vector3 cast(PyObject *obj);
    
template<>
Magnum::Vector3i cast(PyObject *obj);
    
template<>
PyObject* cast(const float &f);
    
template<>
float cast(PyObject *obj);
    
template<>
PyObject* cast(const bool &f);
    
template<>
bool cast(PyObject *obj);
    
template<>
PyObject* cast(const double &f);
    
template<>
double cast(PyObject *obj);

template<>
PyObject* cast(const int &i);
    
template<>
int cast(PyObject *obj);
    
    
/**
 * check if type can be converted
 */
template <typename T>
bool check(PyObject *o);
    
template <>
bool check<bool>(PyObject *o);
    
    
/**
 * grab either the i'th arg from the args, or keywords.
 *
 * gets a reference to the object, NULL if not exist.
 */
PyObject *arg(const char* name, int index, PyObject *_args, PyObject *_kwargs);
        
}

#define MX_BASIC_PYTHON_TYPE_INIT(type) \
HRESULT _Mx ## type ## _Init(PyObject* m) { \
if (PyType_Ready((PyTypeObject*)&Mx ## type ## _Type) < 0) { \
return E_FAIL; \
} \
\
Py_INCREF(&Mx ## type ## _Type); \
if (PyModule_AddObject(m, #type, (PyObject *)&Mx ## type ## _Type) < 0) { \
Py_DECREF(&Mx ## type ## _Type); \
return E_FAIL; \
} \
\
return S_OK;\
}







#endif /* SRC_MDCORE_INCLUDE_MXCONVERT_HPP_ */
