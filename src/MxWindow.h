/*
 * MxWindow.h
 *
 *  Created on: Apr 10, 2020
 *      Author: andy
 */

#ifndef SRC_MXWINDOW_H_
#define SRC_MXWINDOW_H_

#include <mechanica_private.h>

class MxWindow
{
};

/**
 * The the particle type type
 */
CAPI_DATA(PyTypeObject) MxWindow_Type;



/**
 * Init and add to python module
 */
HRESULT MxWindow_init(PyObject *m);

#endif /* SRC_MXWINDOW_H_ */
