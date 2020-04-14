/*
 * MxWindowNative.h
 *
 *  Created on: Apr 13, 2020
 *      Author: andy
 */

#ifndef SRC_MXWINDOWNATIVE_H_
#define SRC_MXWINDOWNATIVE_H_

#include <mechanica_private.h>
#include <GLFW/glfw3.h>




struct MxWindowNative : PyObject
{
};


/**
 * The the particle type type
 */
CAPI_DATA(PyTypeObject) MxWindowNative_Type;



/**
 * Init and add to python module
 */
HRESULT MxWindowNative_init(PyObject *m);

#endif /* SRC_MXWINDOWNATIVE_H_ */
