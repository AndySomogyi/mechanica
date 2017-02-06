project(Mechanica)

cmake_minimum_required (VERSION 3.2)

# Add our module finders to cmake
SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Options for


set(CMAKE_INSTALL_PREFIX "$ENV{HOME}/local/mechanica" CACHE PATH "Install location for Mechancia")


find_package(OpenGL REQUIRED)

find_package(LLVM)

find_package(OpenMM)

add_definitions(-DCX3D_SUPPORT_OPENGL)


include_directories(${OPENGL_INCLUDE_DIR})

if(LLVM_FOUND)
  message("LLVM_FOUND")

  
message("LLVM_CXXFLAGS: ${LLVM_CXXFLAGS}")
message("LLVM_HOST_TARGET: ${LLVM_HOST_TARGET}")

message("LLVM_LDFLAGS: ${LLVM_LDFLAGS}")

message("LLVM_INCLUDE_DIRS: ${LLVM_INCLUDE_DIRS}")



message("LLVM_LIBRARIES: ${LLVM_LIBRARIES}")
message("LLVM_LIBRARY_DIRS: ${LLVM_LIBRARY_DIRS}")
message("LLVM_ROOT_DIR: ${LLVM_ROOT_DIR}")

message("LLVM_VERSION_MAJOR: ${LLVM_VERSION_MAJOR}")
message("LLVM_VERSION_MINOR: ${LLVM_VERSION_MINOR}")
message("LLVM_VERSION_STRING: ${LLVM_VERSION_STRING}")

endif()

find_package(AssImp)

if(ASSIMP_FOUND)
  message("ASSIMP_INCLUDE_DIRS: ${ASSIMP_INCLUDE_DIRS}")
  message("ASSIMP_LIBRARIES: ${ASSIMP_LIBRARIES}")
else()
  message("ASSIMP not found")
endif()





# finds: GLFW_FOUND GLFW_INCLUDE_DIR GLFW_LIBRARIES
# find_package(GLFW 3.2 REQUIRED)


#include_directories(${GLFW_INCLUDE_DIRS})

#message("GLFW_INCLUDE_DIR ${GLFW_INCLUDE_DIR}")

#include_directories(${YOUR_DIRECTORY})

#include_directories(${GLFW_INCLUDE_DIR})

#message("GLFW_LIBRARIES: ${GLFW_LIBRARIES}")

# add  public include dir to include path
include_directories(include)


add_subdirectory(deps)
add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(models)
add_subdirectory(testing)
