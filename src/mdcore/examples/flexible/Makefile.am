
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
AM_LDFLAGS = $(METIS_LDFLAGS) $(FFTW3_LDFLAGS)

# Set-up the library
bin_PROGRAMS = flexible flexible_single

# Plain old tests
# Sources for flexible
flexible_SOURCES = flexible.c
flexible_LDADD = ../../src/.libs/libmdcore.a

# Sources for flexible_single
flexible_single_SOURCES = flexible.c
flexible_single_CFLAGS = -DFPTYPE_SINGLE $(AM_CFLAGS)
flexible_single_LDADD = ../../src/.libs/libmdcore_single.a

