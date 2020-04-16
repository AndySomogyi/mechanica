/*
 * MxPyTest.h
 *
 *  Created on: Apr 16, 2020
 *      Author: andy
 */

#ifndef SRC_MXPYTEST_H_
#define SRC_MXPYTEST_H_

#include "mechanica_private.h"

struct MxPyTest : PyObject
{
};


/**
 * The type object for a MxSymbol.
 */
CAPI_DATA(PyTypeObject) MxPyTest_Type;

HRESULT MxPyTest_init(PyObject *m);



#endif /* SRC_MXPYTEST_H_ */
