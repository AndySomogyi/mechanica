add_executable(
  mx-windowless
  WindowlessTestApp.cpp
  )


target_include_directories(mx-windowless
  PRIVATE
  ${glfw_INCLUDE_DIRS}
  "${GLFW_SOURCE_DIR}/deps"
  )


target_link_libraries(mx-windowless
  mechanica
  Magnum::Magnum
  Magnum::MeshTools
  Magnum::SceneGraph
  Magnum::Shaders
  Magnum::Primitives
  Magnum::DebugTools
  TgaImageConverter
  )


if(MX_APPLE)
  target_link_libraries(mx-windowless MagnumWindowlessCglApplication)
elseif(MX_LINUX)
  target_link_libraries(mx-windowless MagnumWindowlessEglApplication)
elseif(MX_WINDOWS)
  target_link_libraries(mx-windowless MagnumWindowlessWglApplication)
endif()
