add_executable(
  mxinfo
  mxinfo.cpp
  )

target_link_libraries(mxinfo
  mechanica
  Magnum::Magnum
  Magnum::MeshTools
  Magnum::SceneGraph
  Magnum::Shaders
  Magnum::Primitives
  )

if(MX_APPLE)
  target_link_libraries(mxinfo MagnumWindowlessCglApplication)
elseif(MX_LINUX)
  target_link_libraries(mxinfo MagnumWindowlessEglApplication)
elseif(MX_WINDOWS)
  target_link_libraries(mxinfo MagnumWindowlessWglApplication)
endif()
