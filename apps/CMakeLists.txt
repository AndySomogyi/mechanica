add_subdirectory(mxinfo)


add_executable(
  mx-pyrun
  mx-pyrun.cpp
  )

target_link_libraries(mx-pyrun
  Python::Python
  Mechanica::PythonLib
  )
