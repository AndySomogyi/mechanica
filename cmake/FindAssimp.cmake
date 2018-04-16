#.rst:
# Find Assimp
# -------------
#
# Finds the Assimp library. This module defines:
#
#  Assimp_FOUND           - True if Assimp library is found
#  Assimp::Assimp         - Assimp imported target
#
# Additionally these variables are defined for internal usage:
#
#  ASSIMP_LIBRARY         - Assimp library
#  ASSIMP_LIBRARIES       - Same as ASSIMP_LIBRARY
#  ASSIMP_INCLUDE_DIR     - Include dir
#

#
#   This file is part of Magnum.
#
#   Copyright © 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017
#             Vladimír Vondruš <mosra@centrum.cz>
#   Copyright © 2017 Jonathan Hale <squareys@googlemail.com>
#
#   Permission is hereby granted, free of charge, to any person obtaining a
#   copy of this software and associated documentation files (the "Software"),
#   to deal in the Software without restriction, including without limitation
#   the rights to use, copy, modify, merge, publish, distribute, sublicense,
#   and/or sell copies of the Software, and to permit persons to whom the
#   Software is furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included
#   in all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#   DEALINGS IN THE SOFTWARE.
#

if(APPLE)
  # look in the brew directory
  find_path(
    ASSIMP_INCLUDE_DIR
    NAMES assimp/anim.h
    PATHS /usr/local/include/
    )
else()
  find_path(ASSIMP_INCLUDE_DIR NAMES assimp/anim.h HINTS include)
endif()

if(WIN32)
    if(MSVC12)
        set(ASSIMP_MSVC_VERSION "vc120")
    elseif(MSVC14)
        set(ASSIMP_MSVC_VERSION "vc140")
    else()
        message(ERROR "Unsupported MSVC version.")
    endif()

    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(ASSIMP_LIBRARY_DIR "lib64")
    elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
        set(ASSIMP_LIBRARY_DIR "lib32")
    endif()

    find_library(ASSIMP_LIBRARY_RELEASE
      assimp-${ASSIMP_MSVC_VERSION}-mt.lib
      PATHS ${ASSIMP_LIBRARY_DIR})
    
    find_library(ASSIMP_LIBRARY_DEBUG
      assimp-${ASSIMP_MSVC_VERSION}-mtd.lib
      PATHS ${ASSIMP_LIBRARY_DIR})
elseif(APPLE)
    # look for brew's assimp, always get a release build here
    find_library(
      ASSIMP_LIBRARY
      NAMES assimp
      PATHS /usr/local/lib/
      )

    set(ASSIMP_LIBRARY_DEBUG ${ASSIMP_LIBRARY})
    set(ASSIMP_LIBRARY_RELEASE ${ASSIMP_LIBRARY})

    message("ASSIMP_LIBRARY_DEBUG: ${ASSIMP_LIBRARY_DEBUG}")
    message("ASSIMP_LIBRARY_RELEASE: ${ASSIMP_LIBRARY_RELEASE}")

else()
    
    find_library(ASSIMP_LIBRARY_RELEASE libassimp PATHS lib)
    find_library(ASSIMP_LIBRARY_DEBUG libassimpd PATHS lib)
endif()

include(SelectLibraryConfigurations)
select_library_configurations(Assimp)

# first look for the the paths, if we find them, find_package_handle_standard_args
# sets Assimp_FOUND

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Assimp DEFAULT_MSG
    ASSIMP_LIBRARY_DEBUG
    ASSIMP_LIBRARY_RELEASE
    ASSIMP_INCLUDE_DIR)
  
# make the targets if we found Assimp
if(Assimp_FOUND AND NOT TARGET Assimp::Assimp)
    add_library(Assimp::Assimp UNKNOWN IMPORTED)

    # not sure why, but separate IMPORTED_LOCATION_DEBUG and RELEASSE
    # don't work right at least on Unixes when statically linking plugins,
    # on Unixes, juse use a single IMPORTED_LOCATION
    if(WIN32)
      set_target_properties(Assimp::Assimp PROPERTIES
	IMPORTED_LOCATION_DEBUG ${ASSIMP_LIBRARY_DEBUG}
	IMPORTED_LOCATION_RELEASE ${ASSIMP_LIBRARY_RELEASE})
    else()
    set_target_properties(Assimp::Assimp PROPERTIES
      IMPORTED_LOCATION ${ASSIMP_LIBRARY})
    endif()

    set_target_properties(Assimp::Assimp PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES ${ASSIMP_INCLUDE_DIR})

    set(ASSIMP_LIBRARIES ${ASSIMP_LIBRARY})
endif()



