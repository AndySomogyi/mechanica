/*
 * MxWindowProxy.h
 *
 *  Created on: Apr 10, 2020
 *      Author: andy
 */

#ifndef SRC_MXWINDOWPROXY_H_
#define SRC_MXWINDOWPROXY_H_

#include <mechanica_private.h>

class MxWindowProxy
{
};

/**
 * The the particle type type
 */
CAPI_DATA(PyTypeObject) MxWindowProxy_Type;



/**
 * Init and add to python module
 */
HRESULT MxWindowProxy_init(PyObject *m);

#endif /* SRC_MXWINDOWPROXY_H_ */
