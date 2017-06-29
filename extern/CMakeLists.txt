# build glfw here, set up variables so magnum does not try to find it
add_subdirectory(glfw)

set(GLFW_Found YES)

# glfw does not set it's target include directories, but it turns out we
# can set them after the glfw subproject is processed.
# This approach enables us to simply target_link_libraries(MyProgram glfw), and
# all the glfw include directories automatically get added to the MyProgram.
target_include_directories(glfw PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/glfw/include/GLFW>
  )

target_include_directories(glfw PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/glfw/deps>
  )

# magnum likes these names:
#  GLFW::GLFW               - GLFW imported target
#  GLFW_LIBRARY             - GLFW library
#  GLFW_INCLUDE_DIR         - Root include dir
#
add_library(GLFW::GLFW ALIAS glfw)



add_subdirectory(soil)

set(MDCORE_USE_FFTW3 ON CACHE BOOL "Override option" FORCE)
#option(MDCORE_USE_FFTW3 "use FFTW" YES)

add_subdirectory(mdcore)




# Disable building extras we won't need (pure C++ project)
set(NANOGUI_BUILD_PYTHON  OFF CACHE BOOL " " FORCE)
set(NANOGUI_INSTALL       OFF CACHE BOOL " " FORCE)

# Add the configurations from nanogui
#add_subdirectory(nanogui)

# For reliability of parallel build, make the NanoGUI targets dependencies
#set_property(TARGET glfw glfw_objects nanogui PROPERTY FOLDER "dependencies")



add_subdirectory(voro++-0.4.6)

#add_subdirectory(assimp)

# set magnum options
add_subdirectory(corrade)

set(WITH_GLFWAPPLICATION ON CACHE BOOL "build magnum with glfw" FORCE)
set(BUILD_STATIC ON CACHE BOOL "build as static libraries for now..." FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "build as static libraries for now..." FORCE)
set(BUILD_PLUGINS_STATIC ON CACHE BOOL "build as static libraries for now..." FORCE)
set(BUILD_TESTS OFF CACHE BOOL "build as static libraries for now..." FORCE)
set(BUILD_GL_TESTS OFF CACHE BOOL "build as static libraries for now..." FORCE)

# magnum plugins for mesh import testing
set(WITH_ASSIMPIMPORTER    ON CACHE BOOL "build Assimp importer" FORCE)
set(WITH_OBJIMPORTER       ON CACHE BOOL "build Assimp importer" FORCE)          
set(WITH_OPENGEXIMPORTER   ON CACHE BOOL "build Assimp importer" FORCE)
set(WITH_ANYIMAGECONVERTER ON CACHE BOOL "build Assimp importer" FORCE)
set(WITH_AUDIO           ON CACHE BOOL "build Assimp importer" FORCE)
set(WITH_ANYSCENEIMPORTER   ON CACHE BOOL "build Assimp importer" FORCE)

add_subdirectory(magnum)

add_subdirectory(magnum-plugins)
