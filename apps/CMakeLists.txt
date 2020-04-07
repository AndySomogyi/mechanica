add_subdirectory(mxinfo)


add_executable(
  mx-pyrun
  mx-pyrun.cpp
  )

target_link_libraries(mx-pyrun
  Python::Python
  Mechanica::PythonLib
  )
 
 
 



add_executable(mx-glinfo ${PROJECT_SOURCE_DIR}/extern/magnum/src/Magnum/Platform/gl-info.cpp)
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




