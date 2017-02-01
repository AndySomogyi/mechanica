
# LLVM compile options
add_definitions(${LLVM_LDFLAGS})

add_compile_options(-DMX_STRICT)

# Make sure the compiler can find include files for our Hello library
# when other libraries or executables link to Hello
include_directories(PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${OPENMM_INCLUDE_DIR})
include_directories(${ASSIMP_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})

set(SRC
  moduledef.cpp
  mx_callable.cpp
  mx_float.cpp
  mx_import.cpp
  mx_module.cpp
  mx_port.cpp
  mx_string.cpp
  mx_type.cpp
  mx_abstract.cpp
  mx_eval.cpp
  mx_floatobject.cpp
  mx_int.cpp
  mx_number.cpp
  mx_runtime.cpp
  mx_symbol.cpp
  mx_ast.cpp
  mx_expr.cpp
  mx_function.cpp
  mx_list.cpp
  mx_object.cpp
  mx_stmt.cpp
  mx_tupleobject.cpp
  MxMesh.cpp
  MxModule.cpp
  MxSimulator.cpp
  MxTestCube.cpp
  MxTestParticles.cpp
  )


set(HEADERS
  ${MECHANICA_PUBLIC_HEADERS}
  mechanica_llvm.h
  mechanica_private.h
  MxMesh.h
  MxModule.h
  MxSimulator.h
  MxTestCube.h
  MxTestParticles.h
  )


add_library(mechanica ${SRC} ${HEADERS})

source_group("Public Header Files" FILES ${MECHANICA_PUBLIC_HEADERS})

message("MECHANICA_PUBLIC_HEADERS: ${MECHANICA_PUBLIC_HEADERS}")

message("MECHANICA_SRC: ${SRC}")


# Link against LLVM libraries
link_directories(${LLVM_LIBRARY_DIRS})
target_link_libraries(mechanica ${LLVM_LIBRARIES})

target_link_libraries(mechanica ${OPENMM_LIBRARIES_STATIC})
