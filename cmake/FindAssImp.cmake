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


if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(ASSIMP_ARCHITECTURE "64")
elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
  set(ASSIMP_ARCHITECTURE "32")
endif(CMAKE_SIZEOF_VOID_P EQUAL 8)

if(WIN32)
  set(ASSIMP_ROOT_DIR CACHE PATH "ASSIMP root directory")
  
  # Find path of each library
  find_path(ASSIMP_INCLUDE_DIR
    NAMES
    assimp/anim.h
    HINTS
    ${ASSIMP_ROOT_DIR}/include
    )
  
  if(MSVC12)
    set(ASSIMP_MSVC_VERSION "vc120")
  elseif(MSVC14)	
    set(ASSIMP_MSVC_VERSION "vc140")
  endif(MSVC12)
  
  if(MSVC12 OR MSVC14)
    
    find_path(ASSIMP_LIBRARY_DIR
      NAMES
      Assimp-${ASSIMP_MSVC_VERSION}-mt.lib
      HINTS
      ${ASSIMP_ROOT_DIR}/lib${ASSIMP_ARCHITECTURE}
      )
    
    find_library(ASSIMP_LIBRARY_RELEASE				Assimp-${ASSIMP_MSVC_VERSION}-mt.lib 			PATHS ${ASSIMP_LIBRARY_DIR})
    find_library(ASSIMP_LIBRARY_DEBUG				Assimp-${ASSIMP_MSVC_VERSION}-mtd.lib			PATHS ${ASSIMP_LIBRARY_DIR})
		
    set(ASSIMP_LIBRARY 
      optimized 	${ASSIMP_LIBRARY_RELEASE}
      debug		${ASSIMP_LIBRARY_DEBUG}
      )
    
    set(ASSIMP_LIBRARIES "ASSIMP_LIBRARY_RELEASE" "ASSIMP_LIBRARY_DEBUG")
	
    FUNCTION(ASSIMP_COPY_BINARIES TargetDirectory)
      ADD_CUSTOM_TARGET(AssimpCopyBinaries
	COMMAND ${CMAKE_COMMAND} -E copy ${ASSIMP_ROOT_DIR}/bin${ASSIMP_ARCHITECTURE}/assimp-${ASSIMP_MSVC_VERSION}-mtd.dll 	${TargetDirectory}/Debug/assimp-${ASSIMP_MSVC_VERSION}-mtd.dll
	COMMAND ${CMAKE_COMMAND} -E copy ${ASSIMP_ROOT_DIR}/bin${ASSIMP_ARCHITECTURE}/assimp-${ASSIMP_MSVC_VERSION}-mt.dll 		${TargetDirectory}/Release/assimp-${ASSIMP_MSVC_VERSION}-mt.dll
	COMMENT "Copying Assimp binaries to '${TargetDirectory}'"
	VERBATIM)
    ENDFUNCTION(ASSIMP_COPY_BINARIES)
    
  endif()
  
else(WIN32)


  message(STATUS "Looking for ASSIMP...")
  
  find_path(
    Assimp_INCLUDE_DIRS
    NAMES postprocess.h scene.h version.h config.h cimport.h
    PATHS /usr/local/include/
    PATH_SUFFIXES assimp
    )
  
  find_library(
    Assimp_LIBRARIES
    NAMES assimp
    PATHS /usr/local/lib/
    )
  
  if (Assimp_INCLUDE_DIRS AND Assimp_LIBRARIES)
    set(Assimp_FOUND TRUE)
  endif (Assimp_INCLUDE_DIRS AND Assimp_LIBRARIES)
  
  if (Assimp_FOUND)
    if (NOT Assimp_FIND_QUIETLY)
      message(STATUS "Found asset importer library: ${Assimp_LIBRARIES}")
    endif (NOT Assimp_FIND_QUIETLY)
  else (Assimp_FOUND)
    if (Assimp_FIND_REQUIRED)
      message(FATAL_ERROR "Could not find asset importer library")
    endif (Assimp_FIND_REQUIRED)
  endif (Assimp_FOUND)
  
endif(WIN32)


if(Assimp_FOUND AND NOT TARGET Assimp)
  add_library(Assimp UNKNOWN IMPORTED)

#    if(ASSIMP_LIBRARY_DEBUG AND ASSIMP_LIBRARY_RELEASE)
#        set_target_properties(Assimp PROPERTIES
#            IMPORTED_LOCATION_DEBUG ${ASSIMP_LIBRARY_DEBUG}
#            IMPORTED_LOCATION_RELEASE ${ASSIMP_LIBRARY_RELEASE})
#    else()
  set_target_properties(Assimp PROPERTIES
    IMPORTED_LOCATION ${Assimp_LIBRARIES})
#    endif()

  set_target_properties(Assimp PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${Assimp_INCLUDE_DIRS})
endif()


