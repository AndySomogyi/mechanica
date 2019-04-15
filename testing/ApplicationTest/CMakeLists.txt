add_executable(
  mx-apptest
  ApplicationTest.cpp
  )


target_include_directories(mx-apptest
  PRIVATE
  ${glfw_INCLUDE_DIRS}
  "${GLFW_SOURCE_DIR}/deps"
  )


target_link_libraries(mx-apptest
  mechanica
  Magnum::Magnum
  Magnum::MeshTools
  Magnum::SceneGraph
  Magnum::Shaders
  Magnum::Primitives
  Magnum::DebugTools
  )


