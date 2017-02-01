# Find OpenMM library.
#
# Looks for the OpenMM libraries at the default (/usr/local) location 
# or custom location found in the OPENMM_ROOT_DIR environment variable. 
#
# The script defines defines: 
#  OPENMM_FOUND     
#  OPENMM_ROOT_DIR
#  OPENMM_INCLUDE_DIR
#  OPENMM_LIBRARY_DIR
#  OPENMM_LIBRARIES      
#  OPENMM_LIBRARIES_D   - debug version of libraries
#  OPENMM_LIBRARIES_STATIC   - static version of libraries 
#  OPENMM_PLUGIN_DIR
#

# Author: Szilard Pall (pszilard@cbr.su.se)
# Updates: Andy Somogyi (somogyie@indiana.edu)

if(OPENMM_INCLUDE_DIR AND OPENMM_LIBRARY_DIR AND OPENMM_PLUGIN_DIR)
    set(OPENMM_FIND_QUIETLY)
endif()

set(OPENMM_ROOT_DIR "$ENV{OPENMM_ROOT_DIR}" CACHE PATH "OpenMM installation directory")

# search for an 'OpenMM.h' file to find the location of the OpenMM dir. 

if(NOT IS_DIRECTORY ${OPENMM_ROOT_DIR})
  message("searching for openmm dir")

  message("looking in $ENV{HOME}/local/openmm")

  find_path(OPENMM_INCLUDE_DIR 
    "OpenMM.h"
    PATHS
    "$ENV{HOME}/local/openmm"
    "/usr/local/openmm"
    "$ENV{HOME}/local/openmm"
    "$ENV{HOME}/local"
    "$ENV{HOME}"
    "/usr/openmm"
    "/usr"
    PATH_SUFFIXES "include"
    CACHE PATH "OpenMM include directory")    

  get_filename_component(OPENMM_ROOT_DIR "${OPENMM_INCLUDE_DIR}../" DIRECTORY)
  set(OPENMM_ROOT_DIR "${OPENMM_ROOT_DIR}" CACHE PATH "OpenMM installation directory" FORCE)
  
endif()

message("OPENMM_ROOT_DIR: ${OPENMM_ROOT_DIR}")

find_library(OPENMM_LIBRARIES
    NAMES OpenMM
    PATHS "${OPENMM_ROOT_DIR}/lib"
    CACHE STRING "OpenMM libraries")

find_library(OPENMM_LIBRARIES_D
    NAMES OpenMM_d
    PATHS "${OPENMM_ROOT_DIR}/lib"
    CACHE STRING "OpenMM debug libraries")

find_library(OPENMM_LIBRARIES_STATIC
    NAMES OpenMM_static
    PATHS "${OPENMM_ROOT_DIR}/lib"
    CACHE STRING "OpenMM static libraries")

if(OPENMM_LIBRARIES_D AND NOT OPENMM_LIBRARIES)
    set(OPENMM_LIBRARIES ${OPENMM_LIBRARIES_D}
        CACHE STRING "OpenMM libraries" FORCE)
    message(WARNING " Only found debug versions of the OpenMM libraries!")
endif()

get_filename_component(OPENMM_LIBRARY_DIR 
    ${OPENMM_LIBRARIES} 
    PATH
    CACHE STRING "OpenMM library path")


if(NOT IS_DIRECTORY ${OPENMM_ROOT_DIR})
    message(FATAL_ERROR "Could not find OpenMM! Set the OPENMM_ROOT_DIR environment "
    "variable to contain the path of the OpenMM installation.")
endif()

if(NOT IS_DIRECTORY ${OPENMM_LIBRARY_DIR})
    message(FATAL_ERROR "Can't find OpenMM libraries. Check your OpenMM installation!")
endif()

# now we can be sure that we have the library dir
if(IS_DIRECTORY "${OPENMM_LIBRARY_DIR}/plugins")
    get_filename_component(OPENMM_PLUGIN_DIR
        "${OPENMM_LIBRARY_DIR}/plugins"
        ABSOLUTE)
    set(OPENMM_PLUGIN_DIR ${OPENMM_PLUGIN_DIR} CACHE PATH "OpenMM plugins directory")
else()
    message(WARNING "Could not detect the OpenMM plugin directory at the default location (${OPENMM_LIBRARY_DIR}/plugins)."
            "Check your OpenMM installation or set the OPENMM_PLUGIN_DIR environment variable!")
endif()

if(NOT OPENMM_INCLUDE_DIR)
    message(FATAL_ERROR "Can't find OpenMM includes. Check your OpenMM installation!")
endif()

set(OPENMM_ROOT_DIR ${OPENMM_ROOT_DIR} CACHE PATH "OpenMM installation directory")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OPENMM DEFAULT_MSG 
                                    OPENMM_ROOT_DIR
                                    OPENMM_LIBRARIES 
                                    OPENMM_LIBRARY_DIR 
                                    OPENMM_INCLUDE_DIR)

mark_as_advanced(OPENMM_INCLUDE_DIR
    OPENMM_LIBRARIES
    OPENMM_LIBRARIES_D
    OPENMM_LIBRARIES_STATIC
    OPENMM_LIBRARY_DIR)
