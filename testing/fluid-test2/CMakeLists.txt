
# Add module path in case this is project root
if(PROJECT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/../../modules/" ${CMAKE_MODULE_PATH})
endif()

#find_package(Corrade REQUIRED Main)

find_package(Magnum REQUIRED
    GL
    MeshTools
    Primitives
    SceneGraph
    Shaders
    )

find_package(Threads REQUIRED)

option(FLUID_SIMULATION_EXAMPLE_MULTITHREADING
    "Build FluidSimulation example with parallel computation"
    ON)
option(FLUID_SIMULATION_EXAMPLE_USE_TBB
    "Using Intel TBB if FluidSimulation is built with parallel computation enabled"
    OFF)

if(FLUID_SIMULATION_EXAMPLE_MULTITHREADING)
    add_definitions(-DMULTITHREADING)
endif()

if(FLUID_SIMULATION_EXAMPLE_USE_TBB)
    add_definitions(-DUSE_TBB)

    find_package(tbb REQUIRED)
    add_definitions(${TBB_DEFINITIONS})
    include_directories(${TBB_INCLUDE_DIRS})
endif()

set_directory_properties(PROPERTIES CORRADE_USE_PEDANTIC_FLAGS ON)
corrade_add_resource(FluidTest2Resources resources.conf)

include_directories(.)
file(GLOB_RECURSE CPP_FILES *.cpp)

add_executable(fluid-test2 WIN32
    ${CPP_FILES}
    ${FluidTest2Resources})

  target_link_libraries(fluid-test2 PRIVATE
    Mechanica::Mechanica
    Magnum::Application
    Magnum::GL
    Magnum::Magnum
    Magnum::MeshTools
    Magnum::Primitives
    Magnum::SceneGraph
    Magnum::Shaders
    Threads::Threads
    Magnum::GlfwApplication
    )

install(TARGETS fluid-test2 DESTINATION ${MAGNUM_BINARY_INSTALL_DIR})

# Make the executable a default target to build & run in Visual Studio
set_property(DIRECTORY ${PROJECT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT magnum-fluidsimulation3d)
