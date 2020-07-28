# build glfw here, set up variables so magnum does not try to find it
# glfw has to come before carbon (and libsbml)

set(WITH_PNGIMPORTER       OFF CACHE BOOL "build PNG importer"    FORCE)

add_subdirectory(glfw)

# build core carbon lib
add_subdirectory(carbon)

set(PYBIND11_CPP_STANDARD -std=c++14)
add_subdirectory(pybind11)

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

set(GLFW_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/extern/glfw/include/GLFW)
set(GLFW_FOUND TRUE)


message("GLFW_VERSION_MAJOR: ${GLFW_VERSION_MAJOR}")
message("GLFW_VERSION_MINOR: ${GLFW_VERSION_MINOR}")
message("GLFW_VERSION_PATCH: ${GLFW_VERSION_PATCH}")
message("GLFW_VERSION: ${GLFW_VERSION}")


message("GLFW_INCLUDE_DIR: ${GLFW_INCLUDE_DIR}")
message("GLFW_LIBRARY: ${GLFW_LIBRARY}")

if(TARGET glfw)
  message("glfw is a target")
  get_target_property(GLFW_VERSION glfw VERSION)
  message("GLFW_VERSION: ${GLFW_VERSION}")

else()
  message("glfw is NOT a target")
endif()

#add_subdirectory(soil)

#set(MDCORE_USE_FFTW3 ON CACHE BOOL "Override option" FORCE)
#option(MDCORE_USE_FFTW3 "use FFTW" YES)



# Disable building extras we won't need (pure C++ project)
# Add the configurations from nanogui
# add_subdirectory(nanogui)
#set(NANOGUI_BUILD_PYTHON  OFF CACHE BOOL " " FORCE)
#set(NANOGUI_INSTALL       OFF CACHE BOOL " " FORCE)
#set(NANOGUI_BUILD_SHARED  OFF CACHE BOOL " " FORCE)


# For reliability of parallel build, make the NanoGUI targets dependencies
#set_property(TARGET glfw glfw_objects nanogui PROPERTY FOLDER "dependencies")

add_subdirectory(Voroxx)



# magnum / corrade want this
if(WIN32)
  set(BUILD_STATIC_UNIQUE_GLOBALS ON CACHE BOOL
    "Do somethign with symbols to make them work on Windows" FORCE)
endif()

add_subdirectory(corrade)

# set magnum options
set(WITH_GLFWAPPLICATION ON CACHE BOOL "build magnum with glfw" FORCE)
set(BUILD_STATIC ON CACHE BOOL "build as static libraries for now..." FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "build as static libraries for now..." FORCE)
set(BUILD_PLUGINS_STATIC ON CACHE BOOL "build as static libraries for now..." FORCE)
set(BUILD_TESTS OFF CACHE BOOL "build as static libraries for now..." FORCE)
set(BUILD_GL_TESTS OFF CACHE BOOL "build as static libraries for now..." FORCE)

# magnum plugins for mesh import testing
set(WITH_ASSIMPIMPORTER    OFF CACHE BOOL "build Assimp importer" FORCE)
set(WITH_OBJIMPORTER       OFF CACHE BOOL "build Assimp importer" FORCE)
set(WITH_OPENGEXIMPORTER   OFF CACHE BOOL "build Assimp importer" FORCE)
set(WITH_ANYIMAGECONVERTER OFF CACHE BOOL "build Assimp importer" FORCE)
set(WITH_AUDIO             OFF CACHE BOOL "build Assimp importer" FORCE)
set(WITH_ANYSCENEIMPORTER  OFF CACHE BOOL "build Assimp importer" FORCE)
set(WITH_PNGIMPORTER       OFF CACHE BOOL "build PNG importer"    FORCE)


if(APPLE)
  message("building on apple")

  set(WITH_CGLCONTEXT ON CACHE BOOL "Build CglContext library" FORCE)

  set(WITH_WINDOWLESSCGLAPPLICATION ON CACHE BOOL
    "Build WindowlessCglApplication library" FORCE)

elseif(UNIX AND NOT APPLE)
  # for Linux, BSD, Solaris, Minix
  message("building on linux")

  set(WITH_WINDOWLESSEGLAPPLICATION ON CACHE BOOL
    "Build WindowlessEglApplication library" FORCE)

  set(WITH_WINDOWLESSGLXAPPLICATION ON CACHE BOOL
    "Build WindowlessGlxApplication library" FORCE)

elseif(WIN32)

  set(WITH_WINDOWLESSWGLAPPLICATION ON CACHE BOOL
    "Build WindowlessWglApplication library" FORCE)

  set(BUILD_STATIC_UNIQUE_GLOBALS ON CACHE BOOL
    "Do somethign with symbols to make them work on Windows" FORCE)

endif()

set(WITH_TGAIMAGECONVERTER ON CACHE BOOL "Build TGA Image Converter" FORCE)

add_subdirectory(magnum)

# add_subdirectory(magnum-plugins)

# SUNDIALS options
set(BUILD_ARKODE         OFF CACHE BOOL "Build this SUNDIALS feature" FORCE)
set(BUILD_CVODE          ON  CACHE BOOL "Build this SUNDIALS feature" FORCE)
set(BUILD_CVODES         OFF CACHE BOOL "Build this SUNDIALS feature" FORCE)
set(BUILD_IDA            OFF CACHE BOOL "Build this SUNDIALS feature" FORCE)
set(BUILD_IDAS           OFF CACHE BOOL "Build this SUNDIALS feature" FORCE)
set(BUILD_KINSOL         OFF CACHE BOOL "Build this SUNDIALS feature" FORCE)

add_subdirectory(sundials)

add_library(DirectXMath INTERFACE)

target_compile_options(DirectXMath INTERFACE -fms-extensions)

target_include_directories(DirectXMath INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/DirectXMath/Inc/>
  #$<INSTALL_INTERFACE:include/mylib>  # <prefix>/include/mylib
)

#set_property(TARGET mylib APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
#  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/mylib>
#  $<INSTALL_INTERFACE:include/mylib>  # <prefix>/include/mylib
#  )

#set_target_properties(DirectXMath::DirectXMath PROPERTIES
#  INTERFACE_INCLUDE_DIRECTORIES $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/DirectXMath/Inc/>
#  )


set(MAGNUM_PYTHON_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/magnum-bindings/src/python/magnum)

set(magnum_SRCS
    #magnum.cpp
    ${MAGNUM_PYTHON_SOURCE_DIR}/math.cpp
    ${MAGNUM_PYTHON_SOURCE_DIR}/math.matrixfloat.cpp
    ${MAGNUM_PYTHON_SOURCE_DIR}/math.matrixdouble.cpp
    ${MAGNUM_PYTHON_SOURCE_DIR}/math.range.cpp
    ${MAGNUM_PYTHON_SOURCE_DIR}/math.vectorfloat.cpp
    ${MAGNUM_PYTHON_SOURCE_DIR}/math.vectorintegral.cpp)


# this is the "object library" target: compiles the sources only once
# need at least one src file in a project for xcode to build it, xcode
# won't work if any product *only* has an object src
add_library(Magnum_PyBind
  OBJECT
  ${magnum_SRCS}
  )

if(WIN32)
#  target_compile_definitions(Magnum_PyBind PUBLIC /bigobj)
endif()


target_include_directories(Magnum_PyBind PUBLIC
  ${Python_INCLUDE_DIRS}
  ${PYBIND11_INCLUDE_DIR}
  )


set_target_properties(Magnum_PyBind PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN 1
  )

set_target_properties(Magnum_PyBind PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(Magnum_PyBind PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/magnum-bindings/src/python/
)
 
target_include_directories(Magnum_PyBind PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/magnum-bindings/src/
)

target_link_libraries(Magnum_PyBind PUBLIC
  Magnum::Magnum
)

target_link_libraries(Magnum_PyBind PRIVATE Magnum::Magnum ${magnum_LIBS})



#set_target_properties(Magnum_PyBind PROPERTIES
#    FOLDER "lib"
#    OUTPUT_NAME "magnum_pybind"
#    LIBRARY_OUTPUT_DIRECTORY ${output_dir})

