/*
    This file is part of Magnum.

    Original authors — credit is appreciated but not required:

        2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018 —
            Vladimír Vondruš <mosra@centrum.cz>

    This is free and unencumbered software released into the public domain.

    Anyone is free to copy, modify, publish, use, compile, sell, or distribute
    this software, either in source code form or as a compiled binary, for any
    purpose, commercial or non-commercial, and by any means.

    In jurisdictions that recognize copyright laws, the author or authors of
    this software dedicate any and all copyright interest in the software to
    the public domain. We make this dedication for the benefit of the public
    at large and to the detriment of our heirs and successors. We intend this
    dedication to be an overt act of relinquishment in perpetuity of all
    present and future rights to this software under copyright law.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
    IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
    CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include "MxTestView.h"
#include <Magnum/GL/Buffer.h>
#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/Platform/GLContext.h>
#include <Magnum/Shaders/VertexColor.h>
#include <GLFW/glfw3.h>

#include <iostream>

using namespace Magnum;

int testWin(int argc, char** argv) {
    /* Initialize the library */
    if(!glfwInit()) return -1;

    /* Create a windowed mode window and its OpenGL context */
    GLFWwindow* const window = glfwCreateWindow(
        340, 480, "Mechanica Test Window", nullptr, nullptr);
    if(!window) {
        glfwTerminate();
        return -1;
    }

    /* Make the window's context current */
    glfwMakeContextCurrent(window);

    {
        /* Create Magnum context in an isolated scope */
        Platform::GLContext ctx{argc, argv};

        /* Setup the colored triangle */
        using namespace Math::Literals;

        struct TriangleVertex {
            Vector2 position;
            Color3 color;
        };
        const TriangleVertex data[]{
            {{-0.5f, -0.5f}, 0xff0000_rgbf},    /* Left vertex, red color */
            {{ 0.5f, -0.5f}, 0x00ff00_rgbf},    /* Right vertex, green color */
            {{ 0.0f,  0.5f}, 0x0000ff_rgbf}     /* Top vertex, blue color */
        };

        GL::Buffer buffer;
        buffer.setData(data);

        GL::Mesh mesh;
        mesh.setPrimitive(GL::MeshPrimitive::Triangles)
            .setCount(3)
            .addVertexBuffer(buffer, 0,
                Shaders::VertexColor2D::Position{},
                Shaders::VertexColor2D::Color3{});

        Shaders::VertexColor2D shader;

        /* Loop until the user closes the window */
        while(!glfwWindowShouldClose(window)) {

            /* Render here */
            GL::defaultFramebuffer.clear(GL::FramebufferClear::Color);
            mesh.draw(shader);

            /* Swap front and back buffers */
            glfwSwapBuffers(window);

            /* Poll for and process events */
            glfwPollEvents();
        }
    }

    glfwTerminate();
}

PyObject *PyTestWin(PyObject *self, PyObject *a)
{
     char *args[] = {"foo", "bar"};

     testWin(1, args);
    Py_RETURN_NONE;
}

MxTestView::MxTestView(int width, int height) :
        context{nullptr}
{
    std::cout << __PRETTY_FUNCTION__ << std::endl;


    glfwWindowHint(GLFW_RESIZABLE, true);
    glfwWindowHint(GLFW_FOCUSED, true);

    /* Context window hints */
    //glfwWindowHint(GLFW_SAMPLES, configuration.sampleCount());
    //glfwWindowHint(GLFW_SRGB_CAPABLE, configuration.isSRGBCapable());

    //onst Configuration::Flags& flags = configuration.flags();
    //#ifdef GLFW_CONTEXT_NO_ERROR
    //glfwWindowHint(GLFW_CONTEXT_NO_ERROR, flags >= Configuration::Flag::NoError);
    //#endif
    //glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, flags >= Configuration::Flag::Debug);
    //glfwWindowHint(GLFW_STEREO, flags >= Configuration::Flag::Stereo);

    /* Set context version, if requested */
    //if(configuration.version() != GL::Version::None) {
    //    Int major, minor;
    //    std::tie(major, minor) = version(configuration.version());

    //    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, major);
    //    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, minor);

    //    if(configuration.version() >= GL::Version::GL310) {
    //        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, true);
    //        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    //    }
    //}

    glfwWindowHint(GLFW_SAMPLES, 4); // 4x antialiasing
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4); // We want OpenGL 4.1
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make MacOS happy; should not be needed
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // We don't want the old OpenGL


    std::cout << "creating window..." << std::endl;

    /* Set context flags */
    window = glfwCreateWindow(width, height,
                              "Test Window", nullptr, nullptr);


    if(!window) {
        Error() << "Platform::GlfwApplication::tryCreateContext(): cannot create context";
        glfwTerminate();
        return;
    }

    std::cout << "created window" << std::endl;


    glfwSetWindowUserPointer(window, this);

    glfwSetWindowPos(window, 500, 100);

    //arcBall.setWindowSize(configuration.size().x(), configuration.size().y());

    /* Proceed with configuring other stuff that couldn't be done with window
       hints */
    //if(configuration.windowFlags() >= Configuration::WindowFlag::Minimized)
    //    glfwIconifyWindow(window);

    //glfwSetInputMode(window, GLFW_CURSOR, Int(configuration.cursorMode()));

    /* Set callbacks */

    //glfwSetWindowRefreshCallback(window, window_refresh_callback);

    //glfwSetCursorPosCallback(window, cursor_position_callback);

    //glfwSetMouseButtonCallback(window, mouse_button_callback);

    //glfwSetCharCallback(window, char_callback);

    //glfwSetScrollCallback(window, scroll_callback);

    //glfwSetWindowSizeCallback(window, size_callback);


    //glfwSetFramebufferSizeCallback(_window, staticViewportEvent);
    //glfwSetKeyCallback(_window, staticKeyEvent);
    //glfwSetCursorPosCallback(_window, staticMouseMoveEvent);
    //glfwSetMouseButtonCallback(_window, staticMouseEvent);
    //glfwSetScrollCallback(_window, staticMouseScrollEvent);
    //glfwSetCharCallback(_window, staticTextInputEvent);

    glfwMakeContextCurrent(window);

    std::cout << "trying to create context" << std::endl;

    /* Return true if the initialization succeeds */

    char *argv[] = {"foo", "bar"};

    context = new Platform::GLContext{1, argv};

    std::cout << "context OK" << std::endl;

    /* Setup the colored triangle */
    using namespace Math::Literals;

    struct TriangleVertex {
        Vector2 position;
        Color3 color;
    };
    const TriangleVertex data[]{
        {{-0.5f, -0.5f}, 0xff0000_rgbf},    /* Left vertex, red color */
        {{ 0.5f, -0.5f}, 0x00ff00_rgbf},    /* Right vertex, green color */
        {{ 0.0f,  0.5f}, 0x0000ff_rgbf}     /* Top vertex, blue color */
    };

    buffer = new Magnum::GL::Buffer{};


    mesh = new Magnum::GL::Mesh{};

    shader = new Magnum::Shaders::VertexColor2D{};


    buffer->setData(data);


    mesh->setPrimitive(GL::MeshPrimitive::Triangles);
    mesh->setCount(3);
    mesh->addVertexBuffer(*buffer, 0,
            Shaders::VertexColor2D::Position{},
            Shaders::VertexColor2D::Color3{});


    /* Loop until the user closes the window */
//   while(!glfwWindowShouldClose(window)) {
//
//        /* Render here */
//        GL::defaultFramebuffer.clear(GL::FramebufferClear::Color);
//        mesh->draw(*shader);
//
//        /* Swap front and back buffers */
//        glfwSwapBuffers(window);
//
//        /* Poll for and process events */
//        glfwPollEvents();
//    }

    std::cout << "all done creating test window " << std::endl;
}

void MxTestView::draw()
{
    transform = transform * Magnum::Matrix3::rotation(Magnum::Rad(0.05));

    shader->setTransformationProjectionMatrix(transform);
    /* Render here */
    GL::defaultFramebuffer.clear(GL::FramebufferClear::Color);


    mesh->draw(*shader);

    /* Swap front and back buffers */
    glfwSwapBuffers(window);


}

MxTestView::~MxTestView()
{
    delete shader;

    delete mesh;

    delete buffer;

    delete context;

    glfwDestroyWindow(window);
}
