/*
 * MxRenderer.h
 *
 *  Created on: Apr 15, 2020
 *      Author: andy
 */

#ifndef SRC_MXRENDERER_H_
#define SRC_MXRENDERER_H_

#include <mechanica_private.h>

struct MxRenderer : PyObject
{
};



/**
 * The the particle type type
 */
CAPI_DATA(PyTypeObject) MxRenderer_Type;

#endif /* SRC_MXRENDERER_H_ */
