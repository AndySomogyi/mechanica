/*
 * MxSystem.h
 *
 *  Created on: Apr 2, 2017
 *      Author: andy
 */

#ifndef SRC_MXSYSTEM_H_
#define SRC_MXSYSTEM_H_

#include "mechanica_private.h"
#include "MxModel.h"
#include "MxPropagator.h"
#include "MxController.h"
#include "MxView.h"

/**
 * get the python module for the system module
 */
CAPI_FUNC(PyObject*) MxSystem_Module();


HRESULT _MxSystem_init(PyObject *m);

#endif /* SRC_MXSYSTEM_H_ */
