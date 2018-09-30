project(Mechanica)
cmake_minimum_required (VERSION 3.2)

# using magnum, so whole project needs C++11
set(CMAKE_CXX_STANDARD 11)

# Use Magnum's modules to find things, these set up cmake names / aliases using a
# magnum friendly naming scheme, add our module finders to cmake, these find
# stuff mechanica, needs, but magnum does not. Make sure to put Mechanica's
# dir before Magnums, so our versions get picked up *BEFORE* Magnum's.
set(CMAKE_MODULE_PATH
  ${PROJECT_SOURCE_DIR}/cmake
  ${PROJECT_SOURCE_DIR}/extern/magnum/modules
  ${PROJECT_SOURCE_DIR}/extern/magnum-plugins/modules)

message("CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message("CMAKE_OSX_DEPLOYMENT_TARGET: ${CMAKE_OSX_DEPLOYMENT_TARGET}")
message("MACOSX_DEPLOYMENT_TARGET: ${MACOSX_DEPLOYMENT_TARGET}")
message("CMAKE_OSX_SYSROOT: ${CMAKE_OSX_SYSROOT}")
message("XCODE_VERSION: ${XCODE_VERSION}")
message("CMAKE_GENERATOR: ${CMAKE_GENERATOR}")
message("CMAKE_CXX_COMPILER_ID: ${CMAKE_CXX_COMPILER_ID}")
message("CMAKE_CXX_COMPILER_VERSION: ${CMAKE_CXX_COMPILER_VERSION}")


#find_package(GLFW REQUIRED)

find_package(PythonInterp 3.5 REQUIRED)
message("PYTHON_EXECUTABLE: ${PYTHON_EXECUTABLE}") # - path to the Python interpreter
message("PYTHON_VERSION_STRING: ${PYTHON_VERSION_STRING}") # - Python version found e.g. 2.5.2
message("PYTHON_VERSION_MAJOR: ${PYTHON_VERSION_MAJOR}") # - Python major version found e.g. 2
message("PYTHON_VERSION_MINOR: ${PYTHON_VERSION_MINOR}") # - Python minor version found e.g. 5
message("PYTHON_VERSION_PATCH: ${PYTHON_VERSION_PATCH}") # - Python patch version found e.g. 2


find_package(PythonLibs 3.5 REQUIRED)
message("PYTHONLIBS_FOUND: ${PYTHONLIBS_FOUND}")
message("PYTHON_LIBRARIES: ${PYTHON_LIBRARIES}")
message("PYTHON_INCLUDE_DIRS: ${PYTHON_INCLUDE_DIRS}")
message("PYTHONLIBS_VERSION_STRING: ${PYTHONLIBS_VERSION_STRING}")



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

#find_package(LLVM)

#find_package(OpenMM)

add_definitions(-DCX3D_SUPPORT_OPENGL)


#find_package(Bullet)



#if(BULLET_FOUND)
#  message("BULLET_INCLUDE_DIRS: ${BULLET_INCLUDE_DIRS}")
#  message("BULLET_LIBRARIES ${BULLET_LIBRARIES}")
#else()
#  message("BULLET NOT FOUND")
#endif()


include_directories(${OPENGL_INCLUDE_DIR})

#if(LLVM_FOUND)
#  message("LLVM_FOUND")
#  
#  message("LLVM_CXXFLAGS: ${LLVM_CXXFLAGS}")
#  message("LLVM_HOST_TARGET: ${LLVM_HOST_TARGET}")
#  
#  message("LLVM_LDFLAGS: ${LLVM_LDFLAGS}")
#  
#  message("LLVM_INCLUDE_DIRS: ${LLVM_INCLUDE_DIRS}")
#  
#  
#  
#  message("LLVM_LIBRARIES: ${LLVM_LIBRARIES}")
#  message("LLVM_LIBRARY_DIRS: ${LLVM_LIBRARY_DIRS}")
#  message("LLVM_ROOT_DIR: ${LLVM_ROOT_DIR}")
#  
#  message("LLVM_VERSION_MAJOR: ${LLVM_VERSION_MAJOR}")
#  message("LLVM_VERSION_MINOR: ${LLVM_VERSION_MINOR}")
#  message("LLVM_VERSION_STRING: ${LLVM_VERSION_STRING}")
#
#endif()

find_package(Assimp)

if(Assimp_FOUND)
 get_target_property(TMP Assimp::Assimp IMPORTED_LOCATION)
  message("ASSIMP_LIBRARIES: ${TMP}")

  get_target_property(TMP Assimp::Assimp INTERFACE_INCLUDE_DIRECTORIES)


  message("Assimp_INTERFACE_INCLUDE_DIRECTORIES: ${TMP}")


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

message("before extern include path: ${CMAKE_INCLUDE_PATH}")
get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
  message( "dir='${dir}'")
endforeach()

# process dependencies first
add_subdirectory(extern)


message("getting mdcore interface incude:  ${MDCORE_INCLUDE_DIR}")





# add  public include dir to include path
include_directories(include)

add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(models)


# CMake policies: enable MACOSX_RPATH by default
if(POLICY CMP0042)
    cmake_policy(SET CMP0042 NEW)
endif()
# Don't treat imported targets with :: as files
if(POLICY CMP0028)
    cmake_policy(SET CMP0028 NEW)
endif()

find_package(Magnum REQUIRED
  Shaders
  GlfwApplication)


add_subdirectory(testing)
#add_subdirectory(magnum_examples)
