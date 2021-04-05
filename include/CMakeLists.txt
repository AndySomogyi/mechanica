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
  mx_config.h
  mx_runtime.h
  NOMStyle.h
  MxSystem.h
  ${AST_HEADERS}
  ${X3D_HEADERS}
  )

message("_PUBLIC_HEADERS: ${_PUBLIC_HEADERS}")

prepend(PUBLIC_HEADERS  ${_PUBLIC_HEADERS})

set(MECHANICA_PUBLIC_HEADERS ${PUBLIC_HEADERS} PARENT_SCOPE)

set(MX_INCLUDE_DIRS ${MX_INCLUDE_DIRS} PARENT_SCOPE)
