
set(GLAD
  "${GLFW_SOURCE_DIR}/deps/glad/glad.h"
  "${GLFW_SOURCE_DIR}/deps/glad.c"
  )

add_executable(
  MeshTest
  MACOSX_BUNDLE
  AppDelegate.h
  AppDelegate.mm
  MeshTest.h
  MeshTest.cpp
  GrowthModel.h
  GrowthModel.cpp
  main.mm
  MainMenu.xib
  ${GLAD}
  )


target_include_directories(MeshTest
  PRIVATE
  ${glfw_INCLUDE_DIRS}
  "${GLFW_SOURCE_DIR}/deps"
  )


target_link_libraries(MeshTest
  mechanica
  glfw
  Magnum::Magnum
  Magnum::MeshTools
  Magnum::SceneGraph
  Magnum::Shaders
  Magnum::Primitives
  )


set_source_files_properties(
    MainMenu.xib
    PROPERTIES
    MACOSX_PACKAGE_LOCATION
    Resources
)

set_target_properties(
    MeshTest
    PROPERTIES
    MACOSX_BUNDLE_INFO_PLIST
    ${CMAKE_CURRENT_LIST_DIR}/plist.in
)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -framework AppKit")
