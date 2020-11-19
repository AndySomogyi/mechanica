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


namespace mx {

/**
 * convert from c++ to python type
 */
template <typename T>
PyObject *cast(const T& x);
    
/**
 * convert from python to c++ type
 */
template <typename T>
T cast(PyObject *o);
    

template<>
PyObject* cast(const Magnum::Vector3 &v);
    

template<>
Magnum::Vector3 cast(PyObject *obj);
    



    
}







#endif /* SRC_MDCORE_INCLUDE_MXCONVERT_HPP_ */
