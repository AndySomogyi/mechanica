
link_directories(${LLVM_LIBRARY_DIRS})
link_libraries(glfw)
link_libraries(mechanica)
link_libraries(${LLVM_LIBRARIES})
link_libraries(${OPENMM_LIBRARIES_STATIC})
#link_libraries(${OPENGL_LIBRARIES})

include_directories(${GLM_INCLUDE_DIRS} ${glfw_INCLUDE_DIRS} "${GLFW_SOURCE_DIR}/deps")

if (MATH_LIBRARY)
    link_libraries("${MATH_LIBRARY}")
endif()

if (MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

if (GLFW_USE_OSMESA)
    add_definitions(-DUSE_NATIVE_OSMESA)
endif()

if (WIN32)
    set(ICON hmm.rc)
elseif (APPLE)
    set(ICON hmm.icns)
    set_source_files_properties(hmm.icns PROPERTIES
        MACOSX_PACKAGE_LOCATION "Resources")
endif()

set(GLAD "${GLFW_SOURCE_DIR}/deps/glad/glad.h"
  "${GLFW_SOURCE_DIR}/deps/glad.c")

#add_executable(particles2 WIN32 MACOSX_BUNDLE particles.cpp ${ICON} ${GLAD})

add_executable(particles2  particles.cpp ${ICON} ${GLAD})

set(WINDOWS_BINARIES particles2)

#set_target_properties(${WINDOWS_BINARIES} ${CONSOLE_BINARIES} PROPERTIES
#                      FOLDER "GLFW3/Examples")

if (MSVC)
    # Tell MSVC to use main instead of WinMain for Windows subsystem executables
    set_target_properties(${WINDOWS_BINARIES} PROPERTIES
                          LINK_FLAGS "/ENTRY:mainCRTStartup")
endif()



if (0)
  set_target_properties(particles2 PROPERTIES MACOSX_BUNDLE_BUNDLE_NAME "Particles 2")

  
  set_target_properties(${WINDOWS_BINARIES} PROPERTIES
    RESOURCE glfw.icns
    MACOSX_BUNDLE_SHORT_VERSION_STRING ${GLFW_VERSION}
    MACOSX_BUNDLE_LONG_VERSION_STRING ${GLFW_VERSION_FULL}
    MACOSX_BUNDLE_ICON_FILE hmm.icns
    MACOSX_BUNDLE_INFO_PLIST "${GLFW_SOURCE_DIR}/CMake/MacOSXBundleInfo.plist.in")
endif()

