/*
 * MxWindow.h
 *
 *  Created on: Apr 10, 2020
 *      Author: andy
 */

#ifndef SRC_MXWINDOW_H_
#define SRC_MXWINDOW_H_

#include <carbon.h>
#include <mechanica_private.h>
#include <Magnum/Magnum.h>
#include <Magnum/Math/Vector2.h>
#include <Magnum/GL/AbstractFramebuffer.h>
#include <GLFW/glfw3.h>

struct MxWindow : PyObject
{
    
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

    virtual Magnum::Vector2i windowSize() const = 0;
    
    virtual Magnum::GL::AbstractFramebuffer& framebuffer() = 0;
    
    virtual void redraw() = 0;
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
