function(prepend var)
  set(listVar "")
  foreach(f ${ARGN})
                           
    list(APPEND listVar "${CMAKE_CURRENT_SOURCE_DIR}/${f}")
  endforeach(f)
  set(${var} "${listVar}" PARENT_SCOPE)
endfunction(prepend)

set (_PUBLIC_HEADERS
  mx_abstract.h
  mx_import.h
  mx_object.h
  mx_port.h
  mx_runtime.h
  mx_string.h
  mx_tupleobject.h
  mx_number.h
  mx_type.h
  mx_callable.h
  mx_callable.h
  mx_eval.h
  mx_int.h
  mx_float.h
  mx_ast.h
  Mechanica.h
  )

message("_PUBLIC_HEADERS: ${_PUBLIC_HEADERS}")

prepend(PUBLIC_HEADERS  ${_PUBLIC_HEADERS})

set(MECHANICA_PUBLIC_HEADERS ${PUBLIC_HEADERS} PARENT_SCOPE)


