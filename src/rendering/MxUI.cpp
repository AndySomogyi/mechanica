/*
 * MxUI.cpp
 *
 *  Created on: Oct 6, 2018
 *      Author: andy
 */

#include <rendering/MxUI.h>
#include "MxTestView.h"
#include <iostream>

MxTestView *view = nullptr;

void MxUI_init(CObject* m)
{
    std::cout << MX_FUNCTION << std::endl;
}


CAPI_FUNC(HRESULT) MxUI_PollEvents()
{
    glfwPollEvents();
    return S_OK;
}

CAPI_FUNC(HRESULT) MxUI_WaitEvents(double timeout)
{
    glfwWaitEventsTimeout(timeout);
    glfwWaitEvents();
    if(view) {
        view->draw();
    }

    return S_OK;
}

CAPI_FUNC(HRESULT) MxUI_PostEmptyEvent()
{
    glfwPostEmptyEvent();
    return S_OK;
}


PyObject *MxPyUI_PollEvents(PyObject *module)
{
    MxUI_PollEvents();
    if(view) {
        view->draw();
    }
    Py_RETURN_NONE;
}


PyObject* MxPyUI_PostEmptyEvent(PyObject *module)
{
    std::cout << MX_FUNCTION << std::endl;
    std::cout << "MxPyUI_PostEmptyEvent" << std::endl;

    Py_RETURN_NONE;
}

static void error_callback(int error, const char* description)
{
    fprintf(stderr, "Error: %s\n", description);
}


CAPI_FUNC(HRESULT) MxUI_InitializeGraphics(const MxGraphicsConfiguration* conf)
{
    std::cout << MX_FUNCTION << std::endl;

    glfwSetErrorCallback(error_callback);

    if (!glfwInit()) {
        return E_FAIL;
    }


    glfwWindowHint(GLFW_SAMPLES, 4); // 4x antialiasing
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4); // We want OpenGL 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make MacOS happy; should not be needed
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // We don't want the old OpenGL

    return S_OK;
}

PyObject *MxPyUI_InitializeGraphics(PyObject *module, PyObject *args)
{
    MxUI_InitializeGraphics(nullptr);
    Py_RETURN_NONE;
}

PyObject *MxPyUI_CreateTestWindow(PyObject *module, PyObject *args)
{
    std::cout << MX_FUNCTION << std::endl;

    if(!view) {
        view = new MxTestView(500,500);
    }

    Py_RETURN_NONE;
}

PyObject* MxPyUI_WaitEvents(PyObject* module, PyObject* args)
{
    double timeout;

    if (!PyArg_ParseTuple(args, "d", &timeout)) {
        return NULL;
    }

    HRESULT result = MxUI_WaitEvents(timeout);

    Py_RETURN_NONE;
}

PyObject* MxPyUI_DestroyTestWindow(PyObject* module, PyObject* args)
{
    std::cout << MX_FUNCTION << std::endl;

    delete view;
    view = nullptr;
    Py_RETURN_NONE;
}
