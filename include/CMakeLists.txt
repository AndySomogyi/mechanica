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



set (_PUBLIC_HEADERS
  Mechanica.h
  mx_callable.h
  mx_function.h
  mx_list.h
  mx_number.h
  mx_port.h
  mx_string.h
  mx_type.h
  mx_abstract.h
  mx_eval.h
  mx_import.h
  mx_mesh.h
  mx_object.h
  mx_runtime.h
  mx_symbol.h
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
  ${AST_HEADERS}
  ${X3D_HEADERS}
  )

message("_PUBLIC_HEADERS: ${_PUBLIC_HEADERS}")

prepend(PUBLIC_HEADERS  ${_PUBLIC_HEADERS})

set(MECHANICA_PUBLIC_HEADERS ${PUBLIC_HEADERS} PARENT_SCOPE)


