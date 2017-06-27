
link_libraries(glfw)
link_libraries(soil)
link_libraries(${OPENGL_LIBRARIES})


include_directories(${GLM_INCLUDE_DIRS})
include_directories(${glfw_INCLUDE_DIRS} "${GLFW_SOURCE_DIR}/deps")

if (MATH_LIBRARY)
    link_libraries("${MATH_LIBRARY}")
endif()

include_directories(${BULLET_INCLUDE_DIRS})
link_libraries(${BULLET_LIBRARIES})


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
#set(GETOPT "${GLFW_SOURCE_DIR}/deps/getopt.h"
#           "${GLFW_SOURCE_DIR}/deps/getopt.c")
#set(TINYCTHREAD "${GLFW_SOURCE_DIR}/deps/tinycthread.h"
#  "${GLFW_SOURCE_DIR}/deps/tinycthread.c")



set(BLOB_SRC
  blob.cpp
  Shader.h
  Camera.h
  ArcBall.cpp
  ArcBall.hpp
  trackball.c
  trackball.h
  )

# add_executable(blob WIN32 MACOSX_BUNDLE ${BLOB_SRC} ${ICON} ${GLAD})
add_executable(blob ${BLOB_SRC} ${ICON} ${GLAD})


#set_target_properties(${WINDOWS_BINARIES} ${CONSOLE_BINARIES} PROPERTIES
#                      FOLDER "GLFW3/Examples")

if (MSVC)
    # Tell MSVC to use main instead of WinMain for Windows subsystem executables
    set_target_properties(${WINDOWS_BINARIES} PROPERTIES
                          LINK_FLAGS "/ENTRY:mainCRTStartup")
endif()

if (APPLE)
  set_target_properties(blob PROPERTIES MACOSX_BUNDLE_BUNDLE_NAME "Blob")

  
  set_target_properties(blob PROPERTIES
    RESOURCE glfw.icns
    MACOSX_BUNDLE_ICON_FILE hmm.icns
    MACOSX_BUNDLE_INFO_PLIST "${GLFW_SOURCE_DIR}/CMake/MacOSXBundleInfo.plist.in")
endif()

