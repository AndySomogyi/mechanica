/*
 * MxWindowNative.h
 *
 *  Created on: Apr 13, 2020
 *      Author: andy
 */

#ifndef SRC_MXGLFWWINDOW_H_
#define SRC_MXGLFWWINDOW_H_

#include <mechanica_private.h>
#include <GLFW/glfw3.h>





struct MxGlfwWindow : PyObject
{

    // it's a wrapper around a native GLFW window
    GLFWwindow* _window;
};


/**
 * The the particle type type
 */
CAPI_DATA(PyTypeObject) MxGlfwWindow_Type;



/**
 * Init and add to python module
 */
HRESULT MxGlfwWindow_init(PyObject *m);

#endif /* SRC_MXGLFWWINDOW_H_ */
