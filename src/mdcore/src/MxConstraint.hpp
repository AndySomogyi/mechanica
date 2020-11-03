/*
 * MxConstraint.h
 *
 *  Created on: Aug 26, 2020
 *      Author: andy
 */

#ifndef SRC_MDCORE_SRC_MXCONSTRAINT_H_
#define SRC_MDCORE_SRC_MXCONSTRAINT_H_

#include "platform.h"
#include "fptype.h"
#include "carbon.h"
#include <Magnum/Magnum.h>
#include <Magnum/Math/Vector4.h>

struct MxConstraint : PyObject
{
};


CAPI_DATA(PyTypeObject) MxConstraint_Type;

/**
 * internal method, init the Force type, and the forces module and add it to the main module.
 */
HRESULT _MxConstraint_init(PyObject *m);

#endif /* SRC_MDCORE_SRC_MXCONSTRAINT_H_ */
