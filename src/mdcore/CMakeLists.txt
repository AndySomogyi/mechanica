project(mdcore)

cmake_minimum_required(VERSION 3.7)

option(MDCORE_DOUBLE "build double precision (libmdcore_d) version (single precision is always built)" NO)

option(MDCORE_SHARED "build shared library (static library is alwasy built)" NO)

option(MDCORE_USE_MPI "build the MPI version" NO)

option(MDCORE_USE_FFTW3 "use FFTW" NO)

option(MDCORE_USE_OPENMP "used OpenMP (only available with GCC)" NO)

option(SANITIZE "build with -fsanitize=address" NO)

if(SANITIZE)
  add_compile_options(-fsanitize=address)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")
endif()

# define dirs relative to root of mdcore project so it can be
# used as a sub-project
set(MDCORE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(MDCORE_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})


# Add our module finders to cmake  |Demos/StaticBoundaryDemo/CMakeLists.txt:13:     ${GLUT_LIBRARIES}
SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

if(MDCORE_USE_FFTW3)
  find_package(FFTW REQUIRED)
  message("FFTW_LIBRARIES: ${FFTW_LIBRARIES}")
  message("FFTW_INCLUDES: ${FFTW_INCLUDES}")
endif()

# sets the MDCORE_INCLUDE_DIR in the top level cmake,
# so all client products can access include dir. 
set(MDCORE_INCLUDE_DIR
  ${MDCORE_SOURCE_DIR}/include
  ${MDCORE_BINARY_DIR}/include
  CACHE INTERNAL "mdcore public includes" FORCE)
  
# all products access the public header directory
include_directories(${MDCORE_INCLUDE_DIR})

add_subdirectory(src)
add_subdirectory(examples)
