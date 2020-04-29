/*
 * MxPy.h
 *
 *  Created on: Apr 21, 2020
 *      Author: andy
 */

#ifndef SRC_MDCORE_SRC_MXPY_H_
#define SRC_MDCORE_SRC_MXPY_H_

#include "Python.h"

#include <Magnum/Magnum.h>
#include <Magnum/Math/Vector4.h>


enum {
    MxGetSetDef_Error = 0,
    MxGetSetDef_Int,
    MxGetSetDef_Float,
    MxGetSetDef_Double,
    MxGetSetDef_Vector2f,
    MxGetSetDef_Vector3f
};

struct MxGetSetDefInfo {
    uint16_t offset;
    uint16_t kind;
    uint16_t options;
    uint16_t padd;
};

typedef PyObject *(*MxGetSetDefGetter)(PyObject *, MxGetSetDefInfo);
typedef int (*MxGetSetDefSetter)(PyObject *, PyObject *, MxGetSetDefInfo);


struct MxGetSetDef {
    const char *name;
    MxGetSetDefGetter get;
    MxGetSetDefSetter set;
    const char *doc;
    MxGetSetDefInfo info;
};



template<typename T>   // primary template
struct PyGetSetDef_Kind : std::integral_constant<uint16_t, MxGetSetDef_Error> {};

template<>  // explicit specialization for T = int
struct PyGetSetDef_Kind<int> : std::integral_constant<uint16_t, MxGetSetDef_Int> {};

template<>  // explicit specialization for T = int
struct PyGetSetDef_Kind<float> : std::integral_constant<uint16_t, MxGetSetDef_Float> {};

template<>  // explicit specialization for T = int
struct PyGetSetDef_Kind<double> : std::integral_constant<uint16_t, MxGetSetDef_Double> {};

template<>  // explicit specialization for T = Vector3
struct PyGetSetDef_Kind<Magnum::Vector3> : std::integral_constant<uint16_t, MxGetSetDef_Vector3f> {};



PyObject *_MxGetSetDef_Getter(PyObject *obj, MxGetSetDefInfo info);

int _MxGetSetDef_Setter(PyObject *, PyObject *, MxGetSetDefInfo);


template <typename Type, typename Klass>
inline size_t constexpr offset_of(Type Klass::*member) {
    constexpr Klass object {};
    return size_t(&(object.*member)) - size_t(&object);
}

template<typename C, typename T>
PyGetSetDef MakeAttibute(const char* name, const char* doc, T C::*pm) {

    MxGetSetDef result;

    result.doc = doc;
    result.name = name;
    result.info.kind = PyGetSetDef_Kind<T>::value;
    result.info.offset = offset_of(pm);
    result.get = _MxGetSetDef_Getter;
    result.set = _MxGetSetDef_Setter;
    return *((PyGetSetDef*)&result);
}

#include <pybind11/pybind11.h>

template<typename Klass, typename VarType, VarType Klass::*pm>
PyGetSetDef MakeAttibuteGetSet(const char* name, const char* doc) {

    PyGetSetDef result;


    // Convert lambda 'la' to function pointer 'ptr':
    auto la = []( int a ) { return a + 1; };
    int (*ptr)( int ) = la;

    //    The get function takes one PyObject* parameter (the instance) and a
    //    function pointer (the associated closure):
    //
    //    typedef PyObject *(*getter)(PyObject *, void *);
    //    It should return a new reference on success or NULL with a set
    //    exception on failure.
    //
    //    set functions take two PyObject* parameters (the instance and the value to be set) and a function pointer (the associated closure):
    //
    //    typedef int (*setter)(PyObject *, PyObject *, void *);
    //    In case the attribute should be deleted the second parameter is NULL. Should return 0 on success or -1 with a set exception on failure.

    auto get = [](PyObject* s, void *) -> PyObject* {
        Klass *self = (Klass*)s;
        VarType var = self->*pm;
        try {
            pybind11::object obj = pybind11::cast(var);
            pybind11::handle result = obj.inc_ref();
            return result.ptr();
        }
        catch(std::exception &e) {
            PyErr_SetString(PyExc_ValueError, e.what());
            return NULL;
        }
        catch(...) {
            PyErr_SetString(PyExc_ValueError, "Unknown Error");
            return NULL;
        }
    };
    
    auto set = [](PyObject *s, PyObject *obj, void *) -> int {
        Klass *self = (Klass*)s;
        VarType *var = &(self->*pm);
        
        
        try {
            VarType o = pybind11::cast<VarType>(obj);
            *var = o;
            return 0;
        }
        catch(std::exception &e) {
            PyErr_SetString(PyExc_ValueError, e.what());
            return -1;
        }
        catch(...) {
            PyErr_SetString(PyExc_ValueError, "Unknown Error");
            return -1;
        }

    };

    result.doc = doc;
    result.name = name;
    result.get = (getter)get;
    result.set = (setter)set;
    return result;
}



#endif /* SRC_MDCORE_SRC_MXPY_H_ */
