/*
 * MxTestView.h
 *
 *  Created on: Oct 8, 2018
 *      Author: andy
 */

#ifndef SRC_MXTESTVIEW_H_
#define SRC_MXTESTVIEW_H_



#include <Magnum/GL/Buffer.h>
#include <Magnum/DefaultFramebuffer.h>
#include <Magnum/Mesh.h>
#include <Magnum/Math/Vector3.h>
#include <Magnum/Math/Matrix3.h>
#include <Magnum/Platform/GlfwApplication.h>
#include <Magnum/Shaders/VertexColor.h>
#include <Magnum/Primitives/Cube.h>
#include <Magnum/GL/Version.h>
#include <Magnum/GL/Renderer.h>
#include <Magnum/Trade/MeshData3D.h>


#include "Magnum/GL/GL.h"

#include <GLFW/glfw3.h>



#include "Magnum/GL/Version.h"
#include <Magnum/Platform/GLContext.h>
#include <GLFW/glfw3.h>

#include "mechanica_private.h"

struct MxTestView
{
    GLFWwindow* window;

    // loads the opengl extensions
    Magnum::Platform::GLContext *context;

    MxTestView(int width, int height);

    void draw();

    Magnum::GL::Buffer *buffer;


    Magnum::GL::Mesh *mesh;

    Magnum::Shaders::VertexColor2D *shader;

    ~MxTestView();

    Magnum::Matrix3 transform;
};

int testWin(int argc, char** argv);

PyObject *PyTestWin(PyObject *self, PyObject *args);

PyObject *PyMessagePump(PyObject *self, PyObject *args);

#endif /* SRC_MXTESTVIEW_H_ */
