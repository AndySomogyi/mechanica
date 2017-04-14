project(Mechanica)

cmake_minimum_required (VERSION 3.2)

# Add our module finders to cmake
SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)



find_package(PythonLibs REQUIRED)
message("PYTHONLIBS_FOUND: ${PYTHONLIBS_FOUND}")                  
message("PYTHON_LIBRARIES: ${PYTHON_LIBRARIES}")                  
message("PYTHON_INCLUDE_DIRS: ${PYTHON_INCLUDE_DIRS}")            
message("PYTHONLIBS_VERSION_STRING: ${PYTHONLIBS_VERSION_STRING}")


find_package(PythonInterp REQUIRED)
message("PYTHON_EXECUTABLE: ${PYTHON_EXECUTABLE}") # - path to the Python interpreter
message("PYTHON_VERSION_STRING: ${PYTHON_VERSION_STRING}") # - Python version found e.g. 2.5.2
message("PYTHON_VERSION_MAJOR: ${PYTHON_VERSION_MAJOR}") # - Python major version found e.g. 2
message("PYTHON_VERSION_MINOR: ${PYTHON_VERSION_MINOR}") # - Python minor version found e.g. 5
message("PYTHON_VERSION_PATCH: ${PYTHON_VERSION_PATCH}") # - Python patch version found e.g. 2

find_package(NumPy)
message("NUMPY_FOUND: ${NUMPY_FOUND}") # - was NumPy found
message("NUMPY_VERSION: ${NUMPY_VERSION}") # - the version of NumPy found as a string
message("NUMPY_VERSION_MAJOR: ${NUMPY_VERSION_MAJOR}") # - the major version number of NumPy
message("NUMPY_VERSION_MINOR: ${NUMPY_VERSION_MINOR}") # - the minor version number of NumPy
message("NUMPY_VERSION_PATCH: ${NUMPY_VERSION_PATCH}") # - the patch version number of NumPy
message("NUMPY_VERSION_DECIMAL: ${NUMPY_VERSION_DECIMAL}") # - e.g. version 1.6.1 is 10601
message("NUMPY_INCLUDE_DIR: ${NUMPY_INCLUDE_DIR}") # - path to the NumPy include files


set(PYTHON_LIBRARY "" CACHE PATH "path to custom python library")
set(PYTHON_INCLUDE_DIR "" CACHE PATH "path to where a custom Python.h")
mark_as_advanced(CLEAR FORCE PYTHON_LIBRARY)
mark_as_advanced(CLEAR FORCE PYTHON_INCLUDE_DIR)






# Options for
find_package(glm REQUIRED)

message("GLM_VERSION: ${GLM_VERSION}")
message("GLM_INCLUDE_DIRS: ${GLM_INCLUDE_DIRS}")


set(CMAKE_INSTALL_PREFIX "$ENV{HOME}/local/mechanica" CACHE PATH "Install location for Mechancia")


set(BULLET_ROOT "$ENV{HOME}/local" CACHE PATH "Root directory of Bullet Physics engine")




find_package(OpenGL REQUIRED)

find_package(LLVM)

find_package(OpenMM)

add_definitions(-DCX3D_SUPPORT_OPENGL)


find_package(Bullet)



if(BULLET_FOUND)
  message("BULLET_INCLUDE_DIRS: ${BULLET_INCLUDE_DIRS}")
  message("BULLET_LIBRARIES ${BULLET_LIBRARIES}")
else()
  message("BULLET NOT FOUND")
endif()


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

message("before deps include path: ${CMAKE_INCLUDE_PATH}")
get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
  message( "dir='${dir}'")
endforeach()

# process dependencies first
add_subdirectory(deps)


message("getting mdcore interface incude:  ${MDCORE_INCLUDE_DIR}")





# add  public include dir to include path
include_directories(include)

add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(models)
add_subdirectory(testing)
