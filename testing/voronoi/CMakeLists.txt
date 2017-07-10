cmake_minimum_required(VERSION 2.8.12)

# CMake policies: enable MACOSX_RPATH by default
if(POLICY CMP0042)
    cmake_policy(SET CMP0042 NEW)
endif()
# Don't treat imported targets with :: as files
if(POLICY CMP0028)
    cmake_policy(SET CMP0028 NEW)
endif()

find_package(Magnum REQUIRED
  Shaders
  GlfwApplication)

set_directory_properties(PROPERTIES CORRADE_USE_PEDANTIC_FLAGS ON)

add_executable(voronoi
  VoronoiTest.cpp
  SphereShader.cpp
  SphereShader.h
  VoronoiTesselator.cpp
  VoronoiTesselator.h
  )

target_link_libraries(voronoi
  mechanica
  Magnum::GlfwApplication
  Magnum::Magnum
  Magnum::MeshTools
  Magnum::SceneGraph
  Magnum::Shaders
  AnyImageImporter
  AssimpImporter
  ObjImporter
  AnySceneImporter
  OpenGexImporter
  Magnum::Primitives
  voro++)


file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/points.obj
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR})


