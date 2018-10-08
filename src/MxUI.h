/*
 * MxUI.h
 *
 *  Created on: Oct 6, 2018
 *      Author: andy
 */

#ifndef SRC_MXUI_H_
#define SRC_MXUI_H_

#include "mx_ui.h"
#include "mechanica_private.h"


PyObject *MxPyUI_PollEvents(PyObject *module);
PyObject *MxPyUI_WaitEvents(PyObject *module, PyObject *timeout);
PyObject *MxPyUI_PostEmptyEvent(PyObject *module);
PyObject *MxPyUI_InitializeGraphics(PyObject *module, PyObject *args);
PyObject *MxPyUI_CreateTestWindow(PyObject *module, PyObject *args);
PyObject *MxPyUI_DestroyTestWindow(PyObject *module, PyObject *args);

void MxUI_init(MxObject *m);

#endif /* SRC_MXUI_H_ */
