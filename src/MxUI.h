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

PyObject *MxPyUI_PollEvents();
PyObject *MxPyUI_WaitEvents(PyObject *timeout);
PyObject *MxPyUI_PostEmptyEvent();

void MxUI_init(MxObject *m);

#endif /* SRC_MXUI_H_ */
