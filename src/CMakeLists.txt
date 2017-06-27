
# LLVM compile options
#add_definitions(${LLVM_LDFLAGS})

add_compile_options(-DMX_STRICT)

# Make sure the compiler can find include files for our Hello library
# when other libraries or executables link to Hello
include_directories(PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${OPENMM_INCLUDE_DIR})
include_directories(${ASSIMP_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${GLM_INCLUDE_DIRS})
include_directories(${MDCORE_INCLUDE_DIR})
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${NUMPY_INCLUDE_DIR})

message(STATUS "numpy: ${NUMPY_INCLUDE_DIR}")


file(GLOB AST_SRC
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  "ast/*.cpp"
  )

file(GLOB X3D_SRC
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  "x3d/*.cpp"
  )


set(SRC
  moduledef.cpp
  mx_callable.cpp
  mx_float.cpp
  mx_import.cpp
  mx_port.cpp
  mx_string.cpp
  mx_abstract.cpp
  mx_eval.cpp
  mx_floatobject.cpp
  mx_int.cpp
  mx_number.cpp
  mx_runtime.cpp
  mx_ast.cpp
  mx_expr.cpp
  mx_function.cpp
  MxList.cpp
  MxObject.cpp
  mx_stmt.cpp
  mx_tupleobject.cpp
  MxMesh.cpp
  MxModule.cpp
  MxSimulator.cpp
  MxTestCube.cpp
  MxTestParticles.cpp
  MxType.cpp
  MxPropagator.cpp
  MxTestMDCoreParticles.cpp
  MxModule.cpp
  MxModel.cpp
  MxSystem.cpp
  MxSymbol.cpp
  MxPropagator.cpp
  MxParticle.cpp
  MxCompositeParticle.cpp
  MxController.cpp
  MxView.cpp
  MxLattice.cpp
  MxLatticeCluster.cpp
  ${AST_SRC}
  ${X3D_SRC}
  )


set(HEADERS
  ${MECHANICA_PUBLIC_HEADERS}
  mechanica_llvm.h
  mechanica_private.h
  MxObject.h
  MxMesh.h
  MxModule.h
  MxSimulator.h
  MxTestCube.h
  MxTestParticles.h
  MxType.h
  MxTestMDCoreParticles.h
  MxModule.h
  MxModel.h
  MxSystem.h
  MxSymbol.h
  MxPropagator.h
  MxParticle.h
  MxCompositeParticle.h
  MxController.h
  MxView.h
  MxLattice.h
  MxLatticeCluster.h
  )

# build all the source as a CMake 'OBJECT' lib, this gets
# linked into the real libraries below. 
add_library(mechanica_OBJECTS OBJECT ${SRC} ${HEADERS})

# need at least one src file in a project for xcode to build it, xcode
# won't work if any product *only* has an object src
add_library(mechanica
  mechanica.cpp
  $<TARGET_OBJECTS:mechanica_OBJECTS>
  )

add_dependencies(mechanica mechanica_OBJECTS)

target_link_libraries(mechanica glfw)
target_link_libraries(mechanica ${OPENGL_LIBRARIES})

add_library(py_mechanica SHARED
  mechanica.cpp
  $<TARGET_OBJECTS:mechanica_OBJECTS>
  )

add_dependencies(py_mechanica mechanica_OBJECTS)

target_link_libraries(py_mechanica glfw)
target_link_libraries(py_mechanica ${OPENGL_LIBRARIES})
target_link_libraries(py_mechanica ${PYTHON_LIBRARIES})



#target_link_libraries(py_mechanica ${LLVM_LIBRARIES})
set_target_properties(py_mechanica PROPERTIES OUTPUT_NAME "mechanica.so")
set_target_properties(py_mechanica PROPERTIES PREFIX "")
set_target_properties(py_mechanica PROPERTIES SUFFIX "")

source_group("Public Header Files" FILES ${MECHANICA_PUBLIC_HEADERS})

message("MECHANICA_PUBLIC_HEADERS: ${MECHANICA_PUBLIC_HEADERS}")

message("MECHANICA_SRC: ${SRC}")


# Link against LLVM libraries
#link_directories(${LLVM_LIBRARY_DIRS})
#target_link_libraries(mechanica ${LLVM_LIBRARIES})

target_link_libraries(mechanica ${OPENMM_LIBRARIES_STATIC})
target_link_libraries(mechanica mdcore_single)
target_link_libraries(mechanica ${PYTHON_LIBRARIES})
