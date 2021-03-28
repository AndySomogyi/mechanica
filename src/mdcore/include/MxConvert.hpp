/*
 * MxConvert.hpp
 *
 *  Created on: Nov 18, 2020
 *      Author: andy
 */

#ifndef SRC_MDCORE_INCLUDE_MXCONVERT_HPP_
#define SRC_MDCORE_INCLUDE_MXCONVERT_HPP_

#include <platform.h>
#include <CConvert.hpp>
#include <Magnum/Magnum.h>
#include <Magnum/Math/Vector3.h>
#include <Magnum/Math/Vector4.h>
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
PyObject* cast(const Magnum::Vector4 &v);
    
template<>
PyObject* cast(const Magnum::Matrix3 &m);
    
template<>
Magnum::Vector3 cast(PyObject *obj);

template<>
Magnum::Vector4 cast(PyObject *obj);

template<>
Magnum::Vector2 cast(PyObject *obj);
    
template<>
Magnum::Vector3i cast(PyObject *obj);
    
template<>
Magnum::Vector2i cast(PyObject *obj);
    
template<>
inline PyObject* cast(const float &f) {return carbon::cast(f);}
    
template<>
inline float cast(PyObject *obj) { return carbon::cast<float>(obj); };
    
template<>
inline PyObject* cast(const bool &f) {return carbon::cast(f); }
    
template<>
inline bool cast(PyObject *obj) { return carbon::cast<bool>(obj); };
    
template<>
inline PyObject* cast(const double &f) {return carbon::cast(f); }
    
template<>
inline double cast(PyObject *obj) {return carbon::cast<double>(obj); };

template<>
inline PyObject* cast(const int &i) {return carbon::cast(i); }
    
template<>
inline int cast(PyObject *obj) {return carbon::cast<int>(obj); };

template<>
inline PyObject* cast(const int16_t &i)  {return carbon::cast(i); }

template<>
inline PyObject* cast(const uint16_t &i)  {return carbon::cast(i); }

template<>
inline PyObject* cast(const uint32_t &i)  {return carbon::cast(i); }

template<>
inline PyObject* cast(const uint64_t &i)  {return carbon::cast(i); }

template<>
inline int16_t cast(PyObject *o) {return (int16_t)carbon::cast<int>(o);};

template<>
inline uint16_t cast(PyObject *o) {return (uint16_t)carbon::cast<int>(o);};

template<>
inline uint32_t cast(PyObject *o) {return (uint32_t)carbon::cast<int>(o);};

template<>
inline uint64_t cast(PyObject *o) {return (uint64_t)carbon::cast<int>(o);};

template<>
inline PyObject* cast(const std::string &s) {return carbon::cast(s); };

template<>
inline std::string cast(PyObject *o) { return carbon::cast<std::string>(o); };
    
/**
 * check if type can be converted
 */
template <typename T>
bool check(PyObject *o);
    
template <>
inline bool check<bool>(PyObject *o) {return carbon::check<bool>(o); }

template <>
inline bool check<int>(PyObject *o) {return o && PyLong_Check(o) != 0;}

template <>
inline bool check<std::string>(PyObject *o) {return o && PyUnicode_Check(o); };
    
    
/**
 * grab either the i'th arg from the args, or keywords.
 *
 * gets a reference to the object, NULL if not exist.
 */
PyObject *py_arg(const char* name, int index, PyObject *_args, PyObject *_kwargs);

template<typename T>
T arg(const char* name, int index, PyObject *args, PyObject *kwargs) {
    PyObject *value = py_arg(name, index, args, kwargs);
    if(value) {
        return cast<T>(value);
    }
    throw std::runtime_error(std::string("missing argument ") + name);
};

template<>
inline PyObject* arg<PyObject*>(const char* name, int index, PyObject *args, PyObject *kwargs) {
    PyObject *value = py_arg(name, index, args, kwargs);
    if(value) {
        return value;
    }
    throw std::runtime_error(std::string("missing argument ") + name);
};

template<typename T>
T arg(const char* name, int index, PyObject *args, PyObject *kwargs, T deflt) {
    
    PyObject *value = py_arg(name, index, args, kwargs);
    if(value) {
        return cast<T>(value);
    }
    return deflt;
};


        
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
