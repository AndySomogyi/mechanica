add_executable(
  CylinderTest
  MACOSX_BUNDLE
  AppDelegate.h
  AppDelegate.mm
  CylinderTest.h
  CylinderTest.cpp
  CylinderModel.h
  CylinderModel.cpp
  main.mm
  MainMenu.xib
  )


target_include_directories(CylinderTest
  PRIVATE
  ${glfw_INCLUDE_DIRS}
  "${GLFW_SOURCE_DIR}/deps"
  )


target_link_libraries(CylinderTest
  mechanica
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
    CylinderTest
    PROPERTIES
    MACOSX_BUNDLE_INFO_PLIST
    ${CMAKE_CURRENT_LIST_DIR}/plist.in
)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -framework AppKit")
