# - Find the NumPy libraries
# This module finds if NumPy is installed, and sets the following variables
# indicating where it is.
#
# TODO: Update to provide the libraries and paths for linking npymath lib.
#
#  Python_NumPy_FOUND
#  Python_NumPy_INCLUDE_DIRS
#  Python_NumPy_VERSION


cmake_minimum_required(VERSION 3.13)

unset(NUMPY_VERSION)
unset(NUMPY_INCLUDE_DIR)

message("findNumpy, looking for python...")

find_package(Python REQUIRED COMPONENTS Interpreter)

if(NOT ${Python_Interpeter_FOUND})
  message("no python found")
else()
  message("found python")
endif()

message("NUMPY Python_FOUND: ${Python_FOUND}")
message("NUMPY Python_Interpeter_FOUND: ${Python_Interpeter_FOUND}")
message("NUMPY Python_VERSION: ${Python_VERSION}")
message("NUMPY Python_EXECUTABLE: ${Python_EXECUTABLE}")


execute_process(COMMAND "${Python_EXECUTABLE}" "-c"
  "import numpy as n; print(n.__version__); print(n.get_include());"
  RESULT_VARIABLE __result
  OUTPUT_VARIABLE __output
  OUTPUT_STRIP_TRAILING_WHITESPACE)

message("result: ${__result}, output: ${__output}")

if(__result MATCHES 0)
  string(REGEX REPLACE ";" "\\\\;" __values ${__output})
  string(REGEX REPLACE "\r?\n" ";"    __values ${__values})
  list(GET __values 0 Python_NumPy_VERSION)
  list(GET __values 1 Python_NumPy_INCLUDE_DIRS)

  message("Numpy, Python_NumPy_VERSION: ${Python_NumPy_VERSION}")
  message("Numpy, Python_NumPy_INCLUDE_DIRS: ${Python_NumPy_INCLUDE_DIRS}")
  
  string(REGEX MATCH "^([0-9])+\\.([0-9])+\\.([0-9])+" __ver_check "${Python_NumPy_VERSION}")
  
  if(NOT "${__ver_check}" STREQUAL "")
    set(NUMPY_VERSION_MAJOR ${CMAKE_MATCH_1})
    set(NUMPY_VERSION_MINOR ${CMAKE_MATCH_2})
    set(NUMPY_VERSION_PATCH ${CMAKE_MATCH_3})
    math(EXPR NUMPY_VERSION_DECIMAL
      "(${NUMPY_VERSION_MAJOR} * 10000) + (${NUMPY_VERSION_MINOR} * 100) + ${NUMPY_VERSION_PATCH}")
    string(REGEX REPLACE "\\\\" "/"  Python_NumPy_INCLUDE_DIRS ${Python_NumPy_INCLUDE_DIRS})
  else()
    unset(Python_NumPy_VERSION)
    unset(Python_NumPy_INCLUDE_DIRS)
    message(STATUS "Requested NumPy version and include path, but got instead:\n${__output}\n")
  endif()
endif()

 
if(Python_NumPy_INCLUDE_DIRS)
  message("NumPy ver. ${Python_NumPy_VERSION} found (include: ${Python_NumPy_INCLUDE_DIRS})")
  set(Python_NumPy_FOUND TRUE)


  add_library(Python::NumPy INTERFACE IMPORTED)
  set_property(TARGET Python::NumPy
    PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${Python_NumPy_INCLUDE_DIRS}")
  #target_link_libraries(Python::NumPy INTERFACE Python::Module)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NumPy REQUIRED_VARS
  Python_NumPy_INCLUDE_DIRS
  Python_NumPy_VERSION
  Python_NumPy_FOUND
  )
  
unset(__result)
unset(__output)
unset(__error)
unset(_value)
unset(__values)
unset(__ver)
unset(_check)
unset(__error)
unset(_value)

