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

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL MECHANICA_ARRAY_API
#include <numpy/arrayobject.h>


#endif /* SRC_MDCORE_INCLUDE_MXNUMPY_H_ */
