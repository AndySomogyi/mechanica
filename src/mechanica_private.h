/*
 *  mechanica_private.h
 *
 *  Created on: Jul 6, 2015
 *      Author: andy
 *
 * The internal, private header file which actually specifies all of the
 * opaque cayman data structures.
 *
 * This file must never be included before the public cayman.h file,
 * as this ensures that the public api files will never have any dependancy
 * on the internal details.
 */

#ifndef CA_STRICT
#define CA_STRICT
#endif

#ifdef CType
#error CType is macro
#endif

#ifndef _INCLUDED_MECHANICA_H_
#include "Mechanica.h"
#endif

#ifdef CType
#error CType is macro
#endif


#ifndef _INCLUDED_CAYMAN_PRIVATE_H_
#define _INCLUDED_CAYMAN_PRIVATE_H_


#include <Python.h>

#ifdef CType
#error CType is macro
#endif



// Games with importing numpy and setting up function pointers.
// Only the main Mechanica python init module, mechanica.cpp defines
// MX_IMPORTING_NUMPY_ARRAY and calls import_array()
#ifndef MX_IMPORTING_NUMPY_ARRAY
#define NO_IMPORT_ARRAY
#endif
#define PY_ARRAY_UNIQUE_SYMBOL MECHANICA_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
//#include <numpy/arrayobject.h>


#include <assert.h>

#ifdef CType
#error CType is macro
#endif



#ifdef CType
#error CType is macro
#endif



#ifdef CType
#error CType is macro
#endif


#include <cstdint>

#ifdef CType
#error CType is macro
#endif


#include <algorithm>

#ifdef CType
#error CType is macro
#endif





typedef uint32_t uint;
typedef uint16_t ushort;
typedef uint8_t uchar;




/**
 * Initialize the runtime eval modules (builtins, globals)
 */
int initEval();


/**
 * Shutdown the eval modules
 */
int finalizeEval();


/**
 * Initialize the AST module
 */
int _CaAstInit();


#define MX_NOTIMPLEMENTED \
    assert("Not Implemented" && 0);\
    return 0;

#include "mx_error.h"



template <class X, class Y>
inline X* dyn_cast(const Y &o) {
    return (CType_IsSubtype(o->ob_type, X::type())) ? reinterpret_cast<X*>(o) : nullptr;
};

/**
 * modulus for negative numbers
 *
 * General mod for integer or floating point
 *
 * int mod(int x, int divisor)
 * {
 *    int m = x % divisor;
 *    return m + (m < 0 ? divisor : 0);
 * }
 */
template<typename XType, typename DivType> XType mod(XType x, DivType divisor)
{
    return (divisor + (x%divisor)) % divisor;
}

//Returns floor(a/n) (with the division done exactly).
//Let ÷ be mathematical division, and / be C++ division.
//We know
//    a÷b = a/b + f (f is the remainder, not all
//                   divisions have exact Integral results)
//and
//    (a/b)*b + a%b == a (from the standard).
//Together, these imply (through algebraic manipulation):
//    sign(f) == sign(a%b)*sign(b)
//We want the remainder (f) to always be >=0 (by definition of flooredDivision),
//so when sign(f) < 0, we subtract 1 from a/n to make f > 0.
template<typename TA, typename TN>
TA flooredDivision(TA a, TN n) {
    TA q(a/n);
    if ((a%n < 0 && n > 0) || (a%n > 0 && n < 0)) --q;
    return q;
}

//flooredModulo: Modulo function for use in the construction
//looping topologies. The result will always be between 0 and the
//denominator, and will loop in a natural fashion (rather than swapping
//the looping direction over the zero point (as in C++11),
//or being unspecified (as in earlier C++)).
//Returns x such that:
//
//Real a = Real(numerator)
//Real n = Real(denominator)
//Real r = a - n*floor(n/d)
//x = Integral(r)
template<typename TA, typename TN>
TA flooredModulo(TA a, TN n) {
    return a - n * flooredDivision(a, n);
}

template<typename TA, typename TN>
TA loopIndex(TA index, TN range) {
    return mod(index + range, range);
}

/**
 * searches for the item in the container. If the item is found,
 * returns the index, otherwise returns -1.
 */
template<typename Vec, typename Val>
int indexOf(const Vec& vec, const Val& val) {
    int result = std::find(vec.begin(), vec.end(), val) - vec.begin();
    return result < vec.size() ? result : -1;
}

template<typename ContainerType, typename SizeType>
typename ContainerType::value_type wrappedAt(ContainerType &container, SizeType index) {
    SizeType wrappedIndex = loopIndex(index, container.size());
    return container.at(wrappedIndex);
}


#endif /* _INCLUDED_CAYMAN_PRIVATE_H_ */

#ifdef CType
#error CType is macro
#endif

