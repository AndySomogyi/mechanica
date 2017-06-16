
link_libraries(glfw)

include_directories(${glfw_INCLUDE_DIRS} "${GLFW_SOURCE_DIR}/deps")


message("testing glfw, include: ${glfw_INCLUDE_DIRS},  ${GLFW_SOURCE_DIR}/deps")

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
    set(ICON glfw.rc)
elseif (APPLE)
    set(ICON glfw.icns)
    set_source_files_properties(glfw.icns PROPERTIES
        MACOSX_PACKAGE_LOCATION "Resources")
endif()

set(GLAD "${GLFW_SOURCE_DIR}/deps/glad/glad.h"
         "${GLFW_SOURCE_DIR}/deps/glad.c")
set(GETOPT "${GLFW_SOURCE_DIR}/deps/getopt.h"
           "${GLFW_SOURCE_DIR}/deps/getopt.c")
set(TINYCTHREAD "${GLFW_SOURCE_DIR}/deps/tinycthread.h"
                "${GLFW_SOURCE_DIR}/deps/tinycthread.c")

add_executable(test_boing WIN32 MACOSX_BUNDLE boing.c ${ICON} ${GLAD})
add_executable(test_gears WIN32 MACOSX_BUNDLE gears.c ${ICON} ${GLAD})
add_executable(test_heightmap WIN32 MACOSX_BUNDLE heightmap.c ${ICON} ${GLAD})
add_executable(test_offscreen offscreen.c ${ICON} ${GLAD})
add_executable(test_particles WIN32 MACOSX_BUNDLE particles.c ${ICON} ${TINYCTHREAD} ${GETOPT} ${GLAD})
add_executable(test_simple WIN32 MACOSX_BUNDLE simple.c ${ICON} ${GLAD})
add_executable(test_splitview WIN32 MACOSX_BUNDLE splitview.c ${ICON} ${GLAD})
add_executable(test_wave WIN32 MACOSX_BUNDLE wave.c ${ICON} ${GLAD})

target_link_libraries(test_particles "${CMAKE_THREAD_LIBS_INIT}")
if (RT_LIBRARY)
    target_link_libraries(test_particles "${RT_LIBRARY}")
endif()

set(WINDOWS_BINARIES test_boing test_gears test_heightmap test_particles test_simple test_splitview test_wave)
set(CONSOLE_BINARIES test_offscreen)

#set_target_properties(${WINDOWS_BINARIES} ${CONSOLE_BINARIES} PROPERTIES
#                      FOLDER "GLFW3/Examples")

if (MSVC)
    # Tell MSVC to use main instead of WinMain for Windows subsystem executables
    set_target_properties(${WINDOWS_BINARIES} PROPERTIES
                          LINK_FLAGS "/ENTRY:mainCRTStartup")
endif()

if (APPLE)
  set_target_properties(test_boing PROPERTIES MACOSX_BUNDLE_BUNDLE_NAME "Boing")
  set_target_properties(test_gears PROPERTIES MACOSX_BUNDLE_BUNDLE_NAME "Gears")
  set_target_properties(test_heightmap PROPERTIES MACOSX_BUNDLE_BUNDLE_NAME "Heightmap")
  set_target_properties(test_particles PROPERTIES MACOSX_BUNDLE_BUNDLE_NAME "Particles")
  set_target_properties(test_simple PROPERTIES MACOSX_BUNDLE_BUNDLE_NAME "Simple")
  set_target_properties(test_splitview PROPERTIES MACOSX_BUNDLE_BUNDLE_NAME "SplitView")
  set_target_properties(test_wave PROPERTIES MACOSX_BUNDLE_BUNDLE_NAME "Wave")
  
  set_target_properties(${WINDOWS_BINARIES} PROPERTIES
    RESOURCE glfw.icns
    MACOSX_BUNDLE_SHORT_VERSION_STRING ${GLFW_VERSION}
    MACOSX_BUNDLE_LONG_VERSION_STRING ${GLFW_VERSION_FULL}
    MACOSX_BUNDLE_ICON_FILE glfw.icns
    MACOSX_BUNDLE_INFO_PLIST "${GLFW_SOURCE_DIR}/CMake/MacOSXBundleInfo.plist.in")
endif()

