add_library(CylinderComponent SHARED
  CylinderTest.h
  CylinderTest.cpp
  CylinderModel.h
  CylinderModel.cpp
  CylinderComponent.h
  CylinderComponent.cpp
  )


target_link_libraries(CylinderComponent
  mechanica
  glfw
  Magnum::Magnum
  Magnum::MeshTools
  Magnum::SceneGraph
  Magnum::Shaders
  Magnum::Primitives
  )

