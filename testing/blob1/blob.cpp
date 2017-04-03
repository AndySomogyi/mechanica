#include <iostream>

#include <glad/glad.h>
//#define GLFW_INCLUDE_GLCOREARB

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

// Other includes
#include "Shader.h"
#include "ArcBall.hpp"
#include "Camera.h"

extern "C" {
#include "trackball.h"
}

// GLM Mathemtics
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// Window dimensions
const GLuint WIDTH = 800, HEIGHT = 600;


// Function prototypes
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
// Called when the window is resized
void framebuffer_size_callback(GLFWwindow* window, int width, int height);

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);

void window_draw(GLFWwindow* window);

void Do_Movement();

GLuint VBO, VAO;

GLuint program;

// Camera
ArcBall ball(glm::vec3(0,0,0), 0.75);
Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));


bool keys[1024];
GLfloat lastX = 400, lastY = 300;
bool firstMouse = true;

GLfloat deltaTime = 0.0f;
GLfloat lastFrame = 0.0f;

/**
 * quaternians to store rotations.
 */
glm::vec4 curquat, lastquat;

/**
 * previous mouse positions.
 */
float beginx, beginy;


// The MAIN function, from here we start the application and run the game loop
int main()
{
    trackball(glm::value_ptr(curquat), 0.0, 0.0, 0.0, 0.0);
    
    // Init GLFW
    glfwInit();


    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);
    //glfwWindowHint(GLFW_SAMPLES, 4);

    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    // Create a GLFWwindow object that we can use for GLFW's functions
    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "LearnOpenGL", nullptr, nullptr);
    glfwMakeContextCurrent(window);

    // Load all OpenGL functions using the glfw loader function
    // If you use SDL you can use: https://wiki.libsdl.org/SDL_GL_GetProcAddress
    if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)) {
        std::cout << "Failed to initialize OpenGL context" << std::endl;
        return -1;
    }

    // glad populates global constants after loading to indicate,
    // if a certain extension/version is available.
    printf("OpenGL %d.%d\n", GLVersion.major, GLVersion.minor);

    // Set the required callback functions
    glfwSetKeyCallback(window, key_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);



    // Setup some OpenGL options
    //glEnable(GL_DEPTH_TEST);

    
    // Build and compile our shader program

    const GLchar *vertexSrc =
	"#version 330 core                            \n"
	"layout (location = 0) in vec3 position;      \n"
	"layout (location = 1) in vec3 color;         \n"
	"                                             \n"
	"out vec3 ourColor;                           \n"
    "uniform mat4 model;                          \n"
    "uniform mat4 view;                           \n"
    "uniform mat4 projection;                     \n"
	"                                             \n"
	"void main()                                  \n"
	"{                                            \n"
	"    gl_Position =                            \n"
    "        projection * view  * model *         \n"
    "        vec4(position, 1.0f);                \n"
	"    ourColor = color;                        \n"
	"}\n";

    const GLchar *fragmentSrc =
	"#version 330 core                            \n"
	"in vec3 ourColor;                            \n"
	"                                             \n"
	"out vec4 color;                              \n"
	"                                             \n"
	"void main()                                  \n"
	"{                                            \n"
	"    color = vec4(ourColor, 1.0f);            \n"
	"}                                            \n";

    program = shaderFromSrc(vertexSrc, fragmentSrc);

    // Set up vertex data (and buffer(s)) and attribute pointers
    GLfloat vertices[] = {
        // Positions         // Colors
        0.5f, -0.5f, 0.0f,   1.0f, 0.0f, 0.0f,  // Bottom Right
       -0.5f, -0.5f, 0.0f,   0.0f, 1.0f, 0.0f,  // Bottom Left
        0.0f,  0.5f, 0.0f,   0.0f, 0.0f, 1.0f   // Top
    };
    
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    // Bind the Vertex Array Object first, then bind and set vertex buffer(s) and attribute pointer(s).
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (GLvoid*)0);
    glEnableVertexAttribArray(0);
    // Color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0); // Unbind VAO


    // Game loop
    while (!glfwWindowShouldClose(window))
    {
    	// Set frame time
    	        GLfloat currentFrame = glfwGetTime();
    	        deltaTime = currentFrame - lastFrame;
    	        lastFrame = currentFrame;

        // Check if any events have been activiated (key pressed, mouse moved etc.) and call
	    // corresponding response functions
        glfwPollEvents();
	    Do_Movement();
        
        window_draw(window);


        // Swap the screen buffers
        glfwSwapBuffers(window);
    }
    // Properly de-allocate all resources once they've outlived their purpose
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    // Terminate GLFW, clearing any resources allocated by GLFW.
    glfwTerminate();
    return 0;
}

// Moves/alters the camera positions based on user input
void Do_Movement()
{
    /*
    // Camera controls
    if(keys[GLFW_KEY_W])
        camera.ProcessKeyboard(FORWARD, deltaTime);
    if(keys[GLFW_KEY_S])
        camera.ProcessKeyboard(BACKWARD, deltaTime);
    if(keys[GLFW_KEY_A])
        camera.ProcessKeyboard(LEFT, deltaTime);
    if(keys[GLFW_KEY_D])
        camera.ProcessKeyboard(RIGHT, deltaTime);
     */
}

// Is called whenever a key is pressed/released via GLFW
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode)
{
    //cout << key << endl;
    if(key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GL_TRUE);
    if (key >= 0 && key < 1024)
    {
        if(action == GLFW_PRESS)
            keys[key] = true;
        else if(action == GLFW_RELEASE)
            keys[key] = false;	
    }
}

void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
    int state = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
    if (state == GLFW_PRESS)
    {
        ball.drag(glm::vec2(xpos, ypos));
        
        int width, height;
        glfwGetWindowSize(window, &width, &height);
        
        trackball(glm::value_ptr(lastquat),
                  (2.0 * beginx - width) / width,
                  (-height + 2.0 * beginy) / height,
                  (2.0 * xpos - width) / width,
                  (-height + 2.0 * ypos) / height
                  );
        
        add_quats(glm::value_ptr(lastquat), glm::value_ptr(curquat), glm::value_ptr(curquat));
        
        beginx = xpos;
        beginy = ypos;
        
    }

    

    
    /*
    if (!keys[GLFW_KEY_C])
    {
        lastX = xpos;
        lastY = ypos;
    }
    else
    {

        GLfloat xoffset = xpos - lastX;
        GLfloat yoffset = lastY - ypos;  // Reversed since y-coordinates go from bottom to left
    
        lastX = xpos;
        lastY = ypos;

        camera.ProcessMouseMovement(xoffset, yoffset);
    }
     */
}



void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
        
        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);
        beginx = xpos;
        beginy = ypos;
        
        ball.beginDrag(glm::vec2(xpos, ypos));
    }
}


void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    //camera.ProcessMouseScroll(yoffset);
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // Use red to clear the screen
    //glClearColor(1, 0, 0, 1);
    

    // Set the viewport
    glViewport(0, 0, width, height);

    glClear(GL_COLOR_BUFFER_BIT);
    
    window_draw(window);
    
    // Swap the screen buffers
    glfwSwapBuffers(window);
}

void window_draw(GLFWwindow* window)
{
    int width, height;
    glfwGetWindowSize(window, &width, &height);
    
    
    
    // Render
    // Clear the colorbuffer
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    
    // Draw the triangle
    
    glUseProgram(program);
    
    // Create camera transformation
    glm::mat4 view;
    // view = camera.GetViewMatrix();
    //view = ball.getTransformation();
    glm::mat4 projection;
    
    //projection = glm::perspective(camera.Zoom, (float)width/(float)height, 0.1f, 1000.0f);
    
    projection =  glm::perspective(45.0f, (float)width/(float)height, 0.1f, 200.0f);
    
    
    // Calculate the model matrix for each object and pass it to shader before drawing
    glm::mat4 model;
    model = glm::translate(model, glm::vec3(0,0,0));
    GLfloat angle = 20.0f * 0;
    
    view = glm::lookAt(glm::vec3(0.0, 0.0, 5), glm::vec3(0.0, 0.0, 0.0), glm::vec3(0.0, 1.0, 0.0));
    
    //view = view * ball.getTransformation();
    
    GLfloat m[4][4];
    build_rotmatrix(m, glm::value_ptr(curquat));
    
    
    
    model = glm::rotate(model, angle, glm::vec3(1.0f, 0.3f, 0.5f));
    
    //model = model * ball.getTransformation();
    
    model = model * glm::make_mat4(&m[0][0]);
    
    
    
    glm::vec3 pos(0.0f, 0, 10.0f);
    glm::vec3 target(0.0f, 0, 0.0f);
    glm::vec3 up(0.0, 1.0f, 0.0f);
    
    //view = glm::lookAt(pos, target, up);
    //projection = glm::perspective(45.0f, (float)width/(float)height, 0.1f, 1000.0f);
    
    
    
    
    // Get the uniform locations
    GLint modelLoc = glGetUniformLocation(program, "model");
    GLint viewLoc = glGetUniformLocation(program, "view");
    GLint projLoc = glGetUniformLocation(program, "projection");
    // Pass the matrices to the shader
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    
    glBindVertexArray(VAO);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);
    
}
