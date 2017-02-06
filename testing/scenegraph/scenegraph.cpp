//========================================================================
// Simple GLFW example
// Copyright (c) Camilla Löwy <elmindreda@glfw.org>
//
// This software is provided 'as-is', without any express or implied
// warranty. In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you must not
//    claim that you wrote the original software. If you use this software
//    in a product, an acknowledgment in the product documentation would
//    be appreciated but is not required.
//
// 2. Altered source versions must be plainly marked as such, and must not
//    be misrepresented as being the original software.
//
// 3. This notice may not be removed or altered from any source
//    distribution.
//
//========================================================================
//! [code]

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "linmath.h"

#include <stdlib.h>
#include <stdio.h>

#include "Mechanica.h"

#include <iostream>
#include <x3d/CyberX3D.h>
#include "X3DBrowserFunc.h"

using namespace std;
using namespace CyberX3D;

static SceneGraph sceneGraph;
static GLFWwindow* window;

static int mouseButton = MOUSE_BUTTON_NONE;
static int mousePos[2];

static const struct
{
    float x, y;
    float r, g, b;
} vertices[3] =
{
    { -0.6f, -0.4f, 1.f, 0.f, 0.f },
    {  0.6f, -0.4f, 0.f, 1.f, 0.f },
    {   0.f,  0.6f, 0.f, 0.f, 1.f }
};

static const char* vertex_shader_text =
"#version 110\n"
"uniform mat4 MVP;\n"
"attribute vec3 vCol;\n"
"attribute vec2 vPos;\n"
"varying vec3 color;\n"
"void main()\n"
"{\n"
"    gl_Position = MVP * vec4(vPos, 0.0, 1.0);\n"
"    color = vCol;\n"
"}\n";

static const char* fragment_shader_text =
"#version 110\n"
"varying vec3 color;\n"
"void main()\n"
"{\n"
"    gl_FragColor = vec4(color, 1.0);\n"
"}\n";

static void error_callback(int error, const char* description)
{
    fprintf(stderr, "Error: %s\n", description);
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
}

void SetSize(GLFWwindow* window, int width, int height)
{
    UpdateViewport(&sceneGraph, width, height);
}

void MouseMotion(GLFWwindow* window, double xpos, double ypos)
{
    mousePos[0] = (int)xpos;
    mousePos[1] = (int)ypos;
}

void MouseButton(GLFWwindow* window, int button, int action, int mods)
{
    mouseButton = MOUSE_BUTTON_NONE;
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
        mouseButton = MOUSE_BUTTON_LEFT;
}



void UpdateSceneGraph()
{
    if (mouseButton == MOUSE_BUTTON_LEFT) {
        int width, height = 0;
        glfwGetWindowSize(window, &width, &height);
        MoveViewpoint(&sceneGraph, width, height, mousePos[0], mousePos[1]);
        
    }

    sceneGraph.update();
}

void DrawSceneGraph(GLFWwindow* window)
{
    DrawSceneGraph(&sceneGraph, OGL_RENDERING_TEXTURE);
    glfwSwapBuffers(window);
}


int main(int argc, char **argv)
{
    GLuint vertex_buffer, vertex_shader, fragment_shader, program;
    GLint mvp_location, vpos_location, vcol_location;
    

    glfwSetErrorCallback(error_callback);

    if (!glfwInit())
        exit(EXIT_FAILURE);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    window = glfwCreateWindow(640, 480, "Simple example", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    
    
    glfwSetKeyCallback(window, key_callback);
    
    // glutReshapeFunc(SetSize);
    glfwSetWindowSizeCallback (window, SetSize);
    
    // glutMotionFunc(MouseMotion);
    glfwSetCursorPosCallback(window, MouseMotion);
    
    
    // glutPassiveMotionFunc(MouseMotion);
    // glutMouseFunc(MouseButton);
    glfwSetMouseButtonCallback(window, MouseButton);
    
    //glutDisplayFunc(DrawSceneGraph);
    
    glfwSetWindowRefreshCallback(window, DrawSceneGraph);
    
    
    //glutIdleFunc(UpdateSceneGraph);
    


    glfwMakeContextCurrent(window);
    gladLoadGLLoader((GLADloadproc) glfwGetProcAddress);
    glfwSwapInterval(1);

    // NOTE: OpenGL error checks have been omitted for brevity

    vertex_shader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex_shader, 1, &vertex_shader_text, NULL);
    glCompileShader(vertex_shader);

    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment_shader, 1, &fragment_shader_text, NULL);
    glCompileShader(fragment_shader);

    program = glCreateProgram();
    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);
    glLinkProgram(program);


    if (2 <= argc) {
        char *filename = argv[1];
        if (sceneGraph.load(filename) == false) {
            std::cout << "Line Number: " <<  sceneGraph.getParserErrorLineNumber() << std::endl;
            std::cout << "Error Message: " <<  sceneGraph.getParserErrorMessage() << std::endl;
            std::cout << "Error Token: " <<   sceneGraph.getParserErrorToken() << std::endl;
            std::cout << "Error Line: " <<   sceneGraph.getParserErrorLineString() << std::endl;
            std::cout << std::flush;
        }
        sceneGraph.initialize();
        if (sceneGraph.getViewpointNode() == NULL)
            sceneGraph.zoomAllViewpoint();
    }


    while (!glfwWindowShouldClose(window))
    {
        UpdateSceneGraph();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwDestroyWindow(window);

    glfwTerminate();
    exit(EXIT_SUCCESS);
}

//! [code]
