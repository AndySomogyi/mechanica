/*
 * CKeyEvent.h
 *
 *  Created on: Dec 29, 2020
 *      Author: andy
 */

#ifndef EXTERN_CARBON_SRC_CKEYEVENT_HPP_
#define EXTERN_CARBON_SRC_CKEYEVENT_HPP_

#include <CEvent.hpp>
#include <Magnum/Platform/GlfwApplication.h>

HRESULT MxKeyEvent_Invoke(Magnum::Platform::GlfwApplication::KeyEvent &event);

/**
 * adds a python event handler.
 */
PyObject* MxKeyEvent_AddDelegate(PyObject *module, PyObject *args, PyObject *kwargs);

CAPI_DATA(PyTypeObject) MxKeyEvent_Type;


HRESULT _MxKeyEvent_Init(PyObject* m);


#endif /* EXTERN_CARBON_SRC_CKEYEVENT_HPP_ */
