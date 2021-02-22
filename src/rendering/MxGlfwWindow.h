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

#include <Magnum/Magnum.h>
#include <Magnum/Math/Vector2.h>
#include <GLFW/glfw3.h>




/**
 * The GLFWWindow provides a glue to connect generate Mechanica events from glfw events.
 */
struct MxGlfwWindow : MxWindow
{
    /**
     * attach to an existing GLFW Window
     */
    MxGlfwWindow(GLFWwindow *win);

    // it's a wrapper around a native GLFW window
    GLFWwindow* _window;

    float f;

    Magnum::Vector2i windowSize() const override;

    void redraw() override;
    
    void setTitle(const char* title);
    
    Magnum::GL::AbstractFramebuffer &framebuffer() override;
    
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
