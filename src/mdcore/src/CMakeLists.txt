
# This file is part of mdcore.
# Coypright (c) 2010 Pedro Gonnet (gonnet@maths.ox.ac.uk)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Set-up CFLAGS for all targets
#AM_CFLAGS = -g -Wall -std=gnu99 $(SIMD_FLAGS) $(OPENMP_CFLAGS) -Werror -O3 -funroll-loops -DTIMER
#AM_LDFLAGS = -version-info 0:0:0 $(METIS_LDFLAGS)


include_directories(${FFTW_INCLUDES})

link_libraries(${FFTW_LIBRARIES})

set(PUBLIC_HEADERS
  "${MDCORE_SOURCE_DIR}/include/errs.h"
  "${MDCORE_SOURCE_DIR}/include/lock.h"
  "${MDCORE_SOURCE_DIR}/include/MxParticle.h"
  "${MDCORE_SOURCE_DIR}/include/MxPotential.h"
  "${MDCORE_SOURCE_DIR}/include/platform.h"
  "${MDCORE_SOURCE_DIR}/include/space_cell.h"
  "${MDCORE_SOURCE_DIR}/include/task.h"
  "${MDCORE_SOURCE_DIR}/include/queue.h"
  "${MDCORE_SOURCE_DIR}/include/space.h"
  "${MDCORE_SOURCE_DIR}/include/runner.h"
  "${MDCORE_SOURCE_DIR}/include/engine.h"
  "${MDCORE_SOURCE_DIR}/include/bond.h"
  "${MDCORE_SOURCE_DIR}/include/rigid.h"
  "${MDCORE_SOURCE_DIR}/include/angle.h"
  "${MDCORE_SOURCE_DIR}/include/exclusion.h"
  "${MDCORE_SOURCE_DIR}/include/dihedral.h"
  "${MDCORE_SOURCE_DIR}/include/cycle.h"
  "${MDCORE_SOURCE_DIR}/include/fptype.h"
  "${MDCORE_SOURCE_DIR}/include/mdcore_config.h"
  "${MDCORE_SOURCE_DIR}/include/MxPy.h"
  "${MDCORE_SOURCE_DIR}/include/MxNumpy.h"
  "${MDCORE_SOURCE_DIR}/include/MxForce.h"
  "${MDCORE_SOURCE_DIR}/include/MxParticleEvent.h"
  "${MDCORE_SOURCE_DIR}/include/MxReactivePotential.h"
  "${MDCORE_SOURCE_DIR}/include/metrics.h"
  "${MDCORE_SOURCE_DIR}/include/MxConvert.hpp"
  "${MDCORE_SOURCE_DIR}/include/MxParticleList.hpp"
  "${MDCORE_SOURCE_DIR}/include/Vertex.hpp"
  "${MDCORE_SOURCE_DIR}/include/MxBody.hpp"
  "${MDCORE_SOURCE_DIR}/include/MxCuboid.hpp"
  "${MDCORE_SOURCE_DIR}/include/Edge.hpp"
  "${MDCORE_SOURCE_DIR}/include/Polygon.hpp"
  "${MDCORE_SOURCE_DIR}/include/Cell.hpp"
  "${MDCORE_SOURCE_DIR}/include/Flux.hpp"
  "${MDCORE_SOURCE_DIR}/include/xoshiro256plus.h"
  )

set(PRIVATE_HEADERS
  btree.h
  mainpage.h
  smoothing_kernel.hpp
  reader.h
  MxConstraint.hpp
  MxCluster.hpp
  engine_advance.h
  MxSecreteUptake.hpp
  SpaceQueue.hpp
  DissapativeParticleDynamics.hpp
  MxBoundaryConditions.hpp
  potential_eval.hpp
  flux_eval.hpp
  dpd_eval.hpp
  cuboid_eval.hpp
  boundary_eval.hpp
  MxBind.hpp
  )

if(MX_THREADING)
  list(APPEND PRIVATE_HEADERS "MxThreadPool.hpp")
  list(APPEND PRIVATE_HEADERS "MxTaskScheduler.hpp")
endif()

#add_library(${_target} ${ARGN} ${_header_list})
#target_include_directories(${_target} PRIVATE "${MDCORE_SOURCE_DIR}/include")

# List required headers

# Both precisions have the same sources.
SET(SOURCES
  space_cell.cpp
  MxParticle.cpp
  runner.cpp
  engine.cpp
  MxPotential.cpp
  space.cpp
  errs.cpp
  bond.cpp
  angle.cpp
  reader.cpp
  rigid.cpp
  dihedral.cpp
  exclusion.cpp
  engine_io.cpp
  engine_bonded.cpp
  engine_rigid.cpp
  engine_advance.cpp
  runner_dopair.cpp
  queue.cpp
  runner_dosort.cpp
  task.cpp
  runner_verlet.cpp
  MxPy.cpp
  MxForce.cpp
  MxParticleEvent.cpp
  MxReactivePotential.cpp
  MxConstraint.cpp
  MxCluster.cpp
  metrics.cpp
  MxConvert.cpp
  MxParticleList.cpp
  Vertex.cpp
  MxBody.cpp
  MxCuboid.cpp
  Edge.cpp
  Polygon.cpp
  Cell.cpp
  Flux.cpp
  MxSecreteUptake.cpp
  SpaceQueue.cpp
  DissapativeParticleDynamics.cpp
  MxBoundaryConditions.cpp
  MxBind.cpp
  )

if(MDCORE_USE_MPI)
  set(SOURCES ${SOURCES} engine_exchange.c)
endif()

# CUDA sources
#if HAVECUDA
#    SOURCES_CUDA = engine_cuda.cu runner_cuda.cu
#    CUDA_MYFLAGS = -O3 -g -DCPU_TPS=2.67e9 -lineinfo -src-in-ptx --maxrregcount 32 -Xptxas="-v" -Xptxas -dlcm=cg -arch sm_20 -ftz=true -fmad=true -DFPTYPE_SINGLE -DWITH_CUDA
#.cu: runner_cuda.h runner_cuda_main.h
#.cu.o:
#	$(NVCC) -c $(NVCCFLAGS) $(CUDA_CFLAGS) $(CUDA_MYFLAGS) $< -o $@
#.cu.lo:
#	$(top_srcdir)/cudalt.py $@ $(NVCC) -c $(NVCCFLAGS) $(CUDA_CFLAGS) $(CUDA_MYFLAGS) $<
#else
#    SOURCES_CUDA =
#endif


# Build the libmdcore libraries
#lib_LTLIBRARIES = libmdcore.la libmdcore_single.la

# Build a CUDA-enabled version too?
#if HAVECUDA
#lib_LTLIBRARIES += libmdcore_single_cuda.la
#if HAVEMPI
#lib_LTLIBRARIES += libmdcore_single_cuda_mpi.la
#endif
#endif

# Build a MPI-enabled version too?
#if HAVEMPI
#lib_LTLIBRARIES += libmdcore_mpi.la libmdcore_single_mpi.la
#endif

# The FPTYPE_DOUBLE pre-processor symbol enables double precision, if this is
# not set, library defaults to single, see fptype.h

function(add_mdcore_single)
  set(MDCORE_BUILD_SINGLE YES)
  set(MDCORE_CONFIG_FILE "mdcore_single_config.h")
  configure_file("${MDCORE_SOURCE_DIR}/include/mdcore.h.in" "${MDCORE_BINARY_DIR}/include/mdcore_single.h")
  configure_file("${MDCORE_SOURCE_DIR}/include/mdcore_config.h.in" "${MDCORE_BINARY_DIR}/include/mdcore_single_config.h")
#  include_directories(${CMAKE_CURRENT_BINARY_DIR})

  set(SINGLE_HEADERS
    ${PUBLIC_HEADERS}
    ${PRIVATE_HEADERS}
    "${MDCORE_BINARY_DIR}/include/mdcore_single.h"
    "${MDCORE_BINARY_DIR}/include/mdcore_single_config.h"
    )

  message("SINGLE_HEADERS: ${SINGLE_HEADERS}")

  source_group("Public Headers Files" FILES ${PUBLIC_HEADERS})
  source_group("Source Files" FILES ${SOURCES})
  source_group("Header Files" FILES ${SINGLE_HEADERS})

  add_library(mdcore_single OBJECT ${SOURCES} ${SINGLE_HEADERS})

  target_compile_definitions(mdcore_single PUBLIC MDCORE_SINGLE)
  target_compile_definitions(mdcore_single PRIVATE C_BUILDING_DLL)

  set_target_properties(mdcore_single PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN 1
    )

  # hack for getting compile definitions through to eclipse project files,
  # this is NOT needed for the build, just for the clipse project files
  message("setting Eclipse project MDCORE_SINGLE preprocessor definition")
  add_definitions(-DMDCORE_SINGLE)

  # temporarily add add these includes until we get a stable public API
  target_include_directories(mdcore_single PUBLIC ${MDCORE_INCLUDE_DIR}
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Python_INCLUDE_DIRS}
    ${Python_NumPy_INCLUDE_DIRS}
    #$<INSTALL_INTERFACE:include/mylib>  # <prefix>/include/mylib
  )

target_link_libraries(mdcore_single PUBLIC
  Carbon::Carbon
  Threads::Threads
  Magnum::Magnum
  )


message("pybind header: ${PYBIND11_INCLUDE_DIR}")

target_include_directories(mdcore_single PRIVATE ${PYBIND11_INCLUDE_DIR})

message("setting mdcore_single interface includes: ${MDCORE_INCLUDE_DIR}")

endfunction(add_mdcore_single)

# alwasy build the single precision version
add_mdcore_single()





# Set-up for general (double-precision) case
#add_library(mdcore  ${SOURCES} ${PUBLIC_HEADERS})

#libmdcore_la_CFLAGS = -DFPTYPE_DOUBLE $(AM_CFLAGS)

# # Set-up for general (double-precision) MPI-endabled case
# libmdcore_mpi_la_SOURCES = $(SOURCES) $(SOURCES_CUDA)
# libmdcore_mpi_la_CFLAGS = -DWITH_MPI $(AM_CFLAGS)
# libmdcore_mpi_la_SHORTNAME = mpi

# # Set-up for single-precision case
# libmdcore_single_la_SOURCES = $(SOURCES)
# libmdcore_single_la_SHORTNAME = sp
# libmdcore_single_la_CFLAGS = -DFPTYPE_SINGLE $(AM_CFLAGS)

#add_library(mdcore_single  ${SOURCES} ${HEADERS})
#target_compile_definitions(mdcore_single PUBLIC FPTYPE_SINGLE)


# # Set-up for single-precision MPI-enabled case
# libmdcore_single_mpi_la_SOURCES = $(SOURCES)
# libmdcore_single_mpi_la_SHORTNAME = sp_mpi
# libmdcore_single_mpi_la_CFLAGS = -DFPTYPE_SINGLE -DWITH_MPI $(AM_CFLAGS)

# # Set-up for single-precision case with CUDA
# libmdcore_single_cuda_la_SOURCES = $(SOURCES) $(SOURCES_CUDA)
# libmdcore_single_cuda_la_SHORTNAME = cu
# libmdcore_single_cuda_la_CFLAGS = -DFPTYPE_SINGLE $(AM_CFLAGS) -DWITH_CUDA $(CUDA_CFLAGS)

# # Set-up for single-precision case with CUDA and MPI
# libmdcore_single_cuda_mpi_la_SOURCES = $(SOURCES) $(SOURCES_CUDA)
# libmdcore_single_cuda_mpi_la_SHORTNAME = cumpi
# libmdcore_single_cuda_mpi_la_CFLAGS = -DFPTYPE_SINGLE $(AM_CFLAGS) -DWITH_CUDA -DWITH_MPI $(CUDA_CFLAGS)


# # Dummy target to make the Cell/BE version
# if HAVECELL
# libmdcore_cell: *.[ch] spu/runner_spu-embed64.o
# 	make -f Makefile_cell
# 	mv libmdcore_cell.a .libs/
# clean_cell:
# 	make -f Makefile_cell clean
# 	rm -f ppu64_*.[ch]
# 	rm -f .libs/libmdcore_cell.a
# clean-recursive: clean_cell
# all: libmdcore_cell
# endif

# # Show the way...
# SUBDIRS = spu
