cmake_minimum_required (VERSION 3.2)

project(Mechanica)

# Add our module finders to cmake
SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)


find_package(OpenGL REQUIRED)

# finds: GLFW_FOUND GLFW_INCLUDE_DIR GLFW_LIBRARIES
# find_package(GLFW 3.2 REQUIRED)


#include_directories(${GLFW_INCLUDE_DIRS})

#message("GLFW_INCLUDE_DIR ${GLFW_INCLUDE_DIR}")

#include_directories(${YOUR_DIRECTORY})

#include_directories(${GLFW_INCLUDE_DIR})

#message("GLFW_LIBRARIES: ${GLFW_LIBRARIES}")

add_subdirectory(deps)
add_subdirectory(testing)
