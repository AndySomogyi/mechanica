/*
 * MxWindowHost.h
 *
 *  Created on: Apr 10, 2020
 *      Author: andy
 */

#ifndef SRC_MXWINDOWHOST_H_
#define SRC_MXWINDOWHOST_H_

#include "mechanica_private.h"

class MxWindowHost
{
};

/**
 * The the particle type type
 */
CAPI_DATA(PyTypeObject) MxWindowHost_Type;



/**
 * Init and add to python module
 */
HRESULT MxWindowHost_init(PyObject *m);

#endif /* SRC_MXWINDOWHOST_H_ */
