/*
 * MxPy.h
 *
 *  Created on: Apr 21, 2020
 *      Author: andy
 */

#ifndef SRC_MDCORE_SRC_MXPY_H_
#define SRC_MDCORE_SRC_MXPY_H_

#include <c_port.h>
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



std::ostream& operator<<(std::ostream& os, const PyObject *obj);

/**
 * same as python delitemstring, except no error if no key.
 */
bool MxDict_DelItemStringNoErr(PyObject *p, const char *key);


// was added in 3.7, dup it here for python 3.6
#if PY_MAJOR_VERSION == 3 and PY_MINOR_VERSION < 7
    PyObject *PyImport_GetModule(PyObject *name);
#endif



#endif /* SRC_MDCORE_SRC_MXPY_H_ */
