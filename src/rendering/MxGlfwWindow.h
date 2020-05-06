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

    enum MouseButton {
        MouseButton1 = GLFW_MOUSE_BUTTON_1,
        MouseButton2 = GLFW_MOUSE_BUTTON_2,
        MouseButton3 = GLFW_MOUSE_BUTTON_3,
        MouseButton4 = GLFW_MOUSE_BUTTON_4,
        MouseButton5 = GLFW_MOUSE_BUTTON_5,
        MouseButton6 = GLFW_MOUSE_BUTTON_6,
        MouseButton7 = GLFW_MOUSE_BUTTON_7,
        MouseButton8 = GLFW_MOUSE_BUTTON_8,
        MouseButtonLast = GLFW_MOUSE_BUTTON_LAST,
        MouseButtonLeft = GLFW_MOUSE_BUTTON_LEFT,
        MouseButtonRight = GLFW_MOUSE_BUTTON_RIGHT,
        MouseButtonMiddle = GLFW_MOUSE_BUTTON_MIDDLE,
    };

    enum State {
        Release = GLFW_RELEASE,
        Press = GLFW_PRESS,
        Repeat = GLFW_REPEAT
    };

    State getMouseButtonState(MouseButton);

    Magnum::Vector2i framebufferSize() const;

    Magnum::Vector2i windowSize() const;

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
