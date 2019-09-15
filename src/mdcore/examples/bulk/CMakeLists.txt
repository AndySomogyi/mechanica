
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

# Automake stuff
#AUTOMAKE_OPTIONS=gnu

# Add the source directory and debug to CFLAGS
#AM_CFLAGS = -g -I../../src -Wall $(OPENMP_CFLAGS)

# Set the linker flags if metis is present
#AM_LDFLAGS = $(METIS_LDFLAGS) $(FFTW3_LDFLAGS)

# Set-up the library
#bin_PROGRAMS = bulk_noverlet bulk_pseudoverlet

# Build a CUDA-enabled version too?
#if HAVECUDA
#    bin_PROGRAMS += bulk_cuda_noverlet bulk_cuda_verlet
#endif

# Build a MPI-enabled version too?
#if HAVEMPI
#    bin_PROGRAMS += bulk_mpi_noverlet bulk_mpi_pseudoverlet
#endif

# Sources for bulk
#bulk_noverlet_SOURCES = bulk.c
#bulk_noverlet_CFLAGS =  -DFPTYPE_SINGLE $(AM_CFLAGS)
#bulk_noverlet_LDADD = ../../src/.libs/libmdcore_single.a
add_executable(bulk_noverlet bulk.c)
target_link_libraries(bulk_noverlet mdcore_single)




# Using pseudo-Verlet lists
# Sources for bulk_pseudoverlet
#bulk_pseudoverlet_SOURCES = bulk.c
#bulk_pseudoverlet_CFLAGS = -DFPTYPE_SINGLE -DENGINE_FLAGS="engine_flag_verlet_pseudo" $(AM_CFLAGS)
#bulk_pseudoverlet_LDADD = ../../src/.libs/libmdcore_single.a
add_executable(bulk_pseudoverlet bulk.c)
target_compile_definitions(bulk_pseudoverlet PUBLIC ENGINE_FLAGS=engine_flag_verlet_pseudo)
target_link_libraries(bulk_pseudoverlet mdcore_single)

# Using CUDA and Verlet lists
# Sources for bulk_cuda_verlet
#bulk_cuda_verlet_SOURCES = bulk.c
#bulk_cuda_verlet_CFLAGS = -DFPTYPE_SINGLE -DWITH_CUDA -DENGINE_FLAGS="(engine_flag_cuda | engine_flag_verlet_pseudo)" $(AM_CFLAGS)
#bulk_cuda_verlet_LDADD = ../../src/.libs/libmdcore_single_cuda.a $(CUDA_LIBS)


# Using CUDA 
# Sources for bulk_cuda
#bulk_cuda_noverlet_SOURCES = bulk.c
#bulk_cuda_noverlet_CFLAGS = -DFPTYPE_SINGLE -DWITH_CUDA -DENGINE_FLAGS="engine_flag_cuda" $(AM_CFLAGS)
#bulk_cuda_noverlet_LDADD = ../../src/.libs/libmdcore_single_cuda.a $(CUDA_LIBS)


# Using MPI
# Sources for bulk
#bulk_mpi_noverlet_SOURCES = bulk.c
#bulk_mpi_noverlet_CFLAGS =  -DFPTYPE_SINGLE -DWITH_MPI $(AM_CFLAGS)
#bulk_mpi_noverlet_LDADD = ../../src/.libs/libmdcore_single_mpi.a

# Sources for bulk_pseudoverlet
#bulk_mpi_pseudoverlet_SOURCES = bulk.c
#bulk_mpi_pseudoverlet_CFLAGS = -DFPTYPE_SINGLE -DWITH_MPI -DENGINE_FLAGS="engine_flag_verlet_pseudo" $(AM_CFLAGS)
#bulk_mpi_pseudoverlet_LDADD = ../../src/.libs/libmdcore_single_mpi.a

