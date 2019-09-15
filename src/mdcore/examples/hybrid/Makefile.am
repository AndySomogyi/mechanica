
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
AUTOMAKE_OPTIONS=gnu

# Add the source directory and debug to CFLAGS
AM_CFLAGS = -g -I../../src -Wall $(OPENMP_CFLAGS)

# Set the linker flags if metis is present
AM_LDFLAGS = -L. $(MPILIBS) $(METIS_LDFLAGS) $(FFTW3_LDFLAGS)

# Set-up the library
if HAVEMPI
bin_PROGRAMS = hybrid #ppm_test
else
bin_PROGRAMS =
endif

# Sources for ppm_test
ppm_test_SOURCES = ppm_test.c
ppm_test_CFLAGS =  -DFPTYPE_SINGLE -DWITH_MPI -I/usr/lib/openmpi/include $(AM_CFLAGS)
ppm_test_LDADD = ../../src/.libs/libmdcore_single_mpi.a -lppm -lmetis -lmpif77 -lmpif90 -lgfortran

# Sources for hybrid
hybrid_SOURCES = hybrid.c
hybrid_CFLAGS =  -DFPTYPE_SINGLE -DWITH_MPI -I/usr/lib/openmpi/include $(AM_CFLAGS)
hybrid_LDADD = ../../src/.libs/libmdcore_single_mpi.a

