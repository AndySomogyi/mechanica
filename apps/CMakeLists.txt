add_subdirectory(mxinfo)

################################################################################
####  mx-pyrun  ################################################################
################################################################################


# spits out a string like:
# ['/usr/local/Cellar/python/3.7.7/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages']
# need to strip off [', ']...
execute_process(
  COMMAND ${Python_EXECUTABLE} -c "import site; print(site.getsitepackages())"
  OUTPUT_VARIABLE PY_SITEPACKAGES
  )

string(REPLACE "['" ""  PY_SITEPACKAGES ${PY_SITEPACKAGES})
string(REPLACE "']" ""  PY_SITEPACKAGES ${PY_SITEPACKAGES})

string(STRIP ${PY_SITEPACKAGES} PY_SITEPACKAGES)

message("PY_SITEPACKAGES: ${PY_SITEPACKAGES}")


# if python / numpy are in debug mode, or otherwise, numpy is installed from an egg,
# it will have a path that looks like this:
# [
# '',
# '/Users/andy/local/lib/python37.zip',
# '/Users/andy/local/lib/python3.7',
# '/Users/andy/local/lib/python3.7/lib-dynload',
# '/Users/andy/local/lib/python3.7/site-packages',
# '/Users/andy/local/lib/python3.7/site-packages/numpy-1.19.0.dev0+0b9b176-py3.7-macosx-10.13-x86_64.egg']
# need to pull the path from that
# can use the numpy include path string from Find_Pythonm, this looks like:
# '/Users/andy/local/lib/python3.7/site-packages/numpy-1.19.0.dev0+0b9b176-py3.7-macosx-10.13-x86_64.egg/numpy/core/include'
# if this causes issues, in future, might get numpy dir directly as:
# python -c "import numpy; print (numpy.__file__)"
# /Users/andy/local/lib/python3.7/site-packages/numpy-1.19.0.dev0+0b9b176-py3.7-macosx-10.13-x86_64.egg/numpy/__init__.py
string(REPLACE "/numpy/core/include"  "" NUMPY_PYPATH ${Python_NumPy_INCLUDE_DIRS})
message("NUMPY_PYPATH: ${NUMPY_PYPATH}")

configure_file(mx-pyrun.h.in mx-pyrun.h)

add_executable(
  mx-pyrun
  mx-pyrun.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/mx-pyrun.h
  )

target_include_directories(mx-pyrun PRIVATE ${Python_INCLUDE_DIRS})

target_include_directories(mx-pyrun PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

if(WIN32 OR CYGWIN)

  # weird bug with CMake / clang on windows, need to add link dir here
  target_link_directories(mx-pyrun PUBLIC ${Python_LIBRARY_DIRS})

  # Link against the Python shared library on Windows
  # target_link_libraries(mechanica_py PRIVATE Python::Python)
  target_link_libraries(mx-pyrun  ${Python_LIBRARY_RELEASE})
else()
  target_link_libraries(mx-pyrun Python::Python)
endif()

target_link_libraries(mx-pyrun
  Mechanica::PythonLib
  )


################################################################################
####  mx-glinfo  ###############################################################
################################################################################


add_executable(mx-glinfo ${PROJECT_SOURCE_DIR}/extern/magnum/src/Magnum/Platform/gl-info.cpp)

set_target_properties(mx-glinfo PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN 1
  )


target_link_libraries(mx-glinfo PRIVATE MagnumGL)

if(MAGNUM_TARGET_HEADLESS OR CORRADE_TARGET_EMSCRIPTEN OR CORRADE_TARGET_ANDROID)
    target_link_libraries(mx-glinfo PRIVATE MagnumWindowlessEglApplication)
elseif(CORRADE_TARGET_IOS)
    target_link_libraries(mx-glinfo PRIVATE MagnumWindowlessIosApplication)
elseif(CORRADE_TARGET_APPLE)
    target_link_libraries(mx-glinfo PRIVATE MagnumWindowlessCglApplication)
elseif(CORRADE_TARGET_UNIX)
    if(MAGNUM_TARGET_GLES AND NOT MAGNUM_TARGET_DESKTOP_GLES)
        target_link_libraries(mx-glinfo PRIVATE MagnumWindowlessEglApplication)
    else()
        target_link_libraries(mx-glinfo PRIVATE MagnumWindowlessGlxApplication)
    endif()
elseif(CORRADE_TARGET_WINDOWS)
    if(MAGNUM_TARGET_GLES AND NOT MAGNUM_TARGET_DESKTOP_GLES)
        target_link_libraries(mx-glinfo PRIVATE MagnumWindowlessWindowsEglApplication)
    else()
        target_link_libraries(mx-glinfo PRIVATE MagnumWindowlessWglApplication)
    endif()
else()
    message(FATAL_ERROR "mx-glinfo is not available on this platform. Set WITH_GL_INFO to OFF to skip building it.")
endif()

#set_target_properties(mx-glinfo PROPERTIES FOLDER "Magnum/Platform")




