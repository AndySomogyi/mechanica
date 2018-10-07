/*
 * MxUI.cpp
 *
 *  Created on: Oct 6, 2018
 *      Author: andy
 */

#include <MxUI.h>
#include <GLFW/glfw3.h>

#include <iostream>

void MxUI_init(MxObject* m)
{
}

MxAPI_FUNC(HRESULT) MxUI_PollEvents()
{
    glfwPollEvents();
    return S_OK;
}

MxAPI_FUNC(HRESULT) MxUI_WaitEvents(double timeout)
{
    glfwWaitEventsTimeout(timeout);
    glfwWaitEvents();
    return S_OK;
}

MxAPI_FUNC(HRESULT) MxUI_PostEmptyEvent()
{
    glfwPostEmptyEvent();
    return S_OK;
}

PyObject* MxPyUI_PollEvents()
{
    MxUI_PollEvents();
    Py_RETURN_NONE;
}

PyObject* MxPyUI_WaitEvents(PyObject* timeout)
{
    Py_RETURN_NONE;
}

PyObject* MxPyUI_PostEmptyEvent()
{
    //MxUI_PostEmptyEvent();
    std::cout << "MxPyUI_PostEmptyEvent" << std::endl;

    Py_RETURN_NONE;
}
