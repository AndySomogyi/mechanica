/*
 * MxWindowNative.h
 *
 *  Created on: Apr 13, 2020
 *      Author: andy
 */

#ifndef SRC_MXGLFWWINDOW_H_
#define SRC_MXGLFWWINDOW_H_

#include <mechanica_private.h>

#include <rendering/MxWindow.h>
#include <GLFW/glfw3.h>





struct MxGlfwWindow : MxWindow
{

    // it's a wrapper around a native GLFW window
    GLFWwindow* _window;

    float f;
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
