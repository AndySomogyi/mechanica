function(prepend var)
  set(listVar "")
  foreach(f ${ARGN})
    list(APPEND listVar "${CMAKE_CURRENT_SOURCE_DIR}/${f}")
  endforeach(f)
  set(${var} "${listVar}" PARENT_SCOPE)
endfunction(prepend)

file(GLOB AST_HEADERS
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  "ast/*.h"
  )

file(GLOB X3D_HEADERS
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  "x3d/*.h"
  )


message("configuring file ... ")
configure_file(mx_config.h.in mx_config.h)

set(MX_INCLUDE_DIRS
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}
  )

message("mx config file: ${CMAKE_CURRENT_BINARY_DIR}/mx_config.h")

set (_PUBLIC_HEADERS
  Mechanica.h
  mx_object.hpp
  mx_callable.h
  mx_function.h
  mx_list.h
  mx_number.h
  mx_abstract.h
  mx_eval.h
  mx_import.h
  mx_mesh.h
  mx_runtime.h
  mx_ast.h
  mx_float.h
  mx_int.h
  mx_module.h
  mx_parser.h
  mx_simulator.h
  mx_tupleobject.h
  mx_particles.h
  mx_particle.h
  mx_composite_particle.h
  mx_propagator.h
  mx_lattice.h
  mx_ui.h
  ${AST_HEADERS}
  ${X3D_HEADERS}
  )

message("_PUBLIC_HEADERS: ${_PUBLIC_HEADERS}")

prepend(PUBLIC_HEADERS  ${_PUBLIC_HEADERS})

set(MECHANICA_PUBLIC_HEADERS ${PUBLIC_HEADERS} PARENT_SCOPE)

set(MX_INCLUDE_DIRS ${MX_INCLUDE_DIRS} PARENT_SCOPE)
