



set(voro++_SOURCE
  cell.cc
  common.cc
  container.cc
  unitcell.cc
  v_compute.cc
  c_loops.cc 
  v_base.cc
  wall.cc
  pre_container.cc
  container_prd.cc
  )


set(voro++_HEADERS
  c_loops.hh
  common.hh
  container.hh
  pre_container.hh
  unitcell.hh
  v_compute.hh
  wall.hh
  cell.hh
  config.hh
  container_prd.hh
  rad_option.hh
  v_base.hh
  voro++.hh
  worklist.hh
  )

add_library(voro++ STATIC
  ${voro++_SOURCE}
  ${voro++_HEADERS})


# TODO: fix this, voro++ as downloaded dumps everything into the same directory
target_include_directories(voro++ PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
#  $<INSTALL_INTERFACE:include/mylib>  # <prefix>/include/mylib
)
