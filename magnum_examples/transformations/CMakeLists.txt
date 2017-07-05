
cmake_minimum_required(VERSION 2.8.12)

# CMake policies: enable MACOSX_RPATH by default
if(POLICY CMP0042)
    cmake_policy(SET CMP0042 NEW)
endif()
# Don't treat imported targets with :: as files
if(POLICY CMP0028)
    cmake_policy(SET CMP0028 NEW)
endif()

find_package(
  Magnum REQUIRED
  Shaders
  GlfwApplication)

find_package(
  Corrade REQUIRED Utility)


set_directory_properties(PROPERTIES CORRADE_USE_PEDANTIC_FLAGS ON)

corrade_add_resource(transformation_resources resources.conf)

add_executable(transformation_3d transformation_3d.cpp ${transformation_resources})

target_link_libraries(transformation_3d
  Magnum::Application
  Magnum::Magnum
  Magnum::Shaders
  PngImporter)

add_executable(transformation_exercise_1 transformation_exercise_1.cpp ${transformation_resources})

target_link_libraries(transformation_exercise_1
  Magnum::Application
  Magnum::Magnum
  Magnum::Shaders
  PngImporter)

add_executable(transformation_exercise_2 transformation_exercise_2.cpp ${transformation_resources})

target_link_libraries(transformation_exercise_2
  Magnum::Application
  Magnum::Magnum
  Magnum::Shaders
  PngImporter)

add_executable(transformation transformation.cpp ${transformation_resources})

target_link_libraries(transformation
  Magnum::Application
  Magnum::Magnum
  Magnum::Shaders
  PngImporter)


