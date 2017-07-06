
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

add_executable(geometry-shader geometry-shader.cpp)

target_link_libraries(geometry-shader
  Magnum::Application
  Magnum::Magnum
  Magnum::Shaders)
