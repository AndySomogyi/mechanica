/*
 * MxNumpy.h
 *
 *  Created on: Mar 20, 2020
 *      Author: andy
 *
 * Deal with intricacies of using the numpy api across multiple files.
 *
 * @see https://docs.scipy.org/doc/numpy/reference/c-api.array.html?highlight=import_array
 */

#ifndef SRC_MDCORE_INCLUDE_MXNUMPY_H_
#define SRC_MDCORE_INCLUDE_MXNUMPY_H_

// Games with importing numpy and setting up function pointers.
// Only the main Mechanica python init module, mechanica.cpp defines
// MX_IMPORTING_NUMPY_ARRAY and calls import_array()
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL MECHANICA_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "numpy/arrayobject.h"


#endif /* SRC_MDCORE_INCLUDE_MXNUMPY_H_ */
