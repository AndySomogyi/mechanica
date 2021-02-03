# Coypright (c) 2020 Andy Somogyi (somogyie at indiana dot edu)
# this is a port of the HOOMMD unit cell code
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
#
# original Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

""" Define lattices.

:py:mod:`mechanica.lattice` provides a general interface to define lattices to initialize systems.

"""

import numpy
import math
from collections import namedtuple

if __name__ == 'mechanica.lattice':
    from . import _mechanica as m
else:
    import mechanica as m


# Multiply two quaternions
# Apply quaternion multiplication per
# http://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
# (requires numpy)
# \param q1 quaternion
# \param q2 quaternion
# \returns q1*q2
def _quatMult(q1, q2):
    s = q1[0]
    v = q1[1:]
    t = q2[0]
    w = q2[1:]
    q = numpy.empty((4,), dtype=numpy.float64)
    q[0] = s*t - numpy.dot(v, w)
    q[1:] = s*w + t*v + numpy.cross(v,w)
    return q

# Rotate a vector by a unit quaternion
# Quaternion rotation per
# http://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
# (requires numpy)
# \param q rotation quaternion
# \param v 3d vector to be rotated
# \returns q*v*q^{-1}
def _quatRot(q, v):
    v = numpy.asarray(v)
    q = numpy.asarray(q)
    # assume q is a unit quaternion
    w = q[0]
    r = q[1:]
    vnew = numpy.empty((3,), dtype=v.dtype)
    vnew = v + 2*numpy.cross(r, numpy.cross(r,v) + w*v)
    return vnew

# make a types vector of the requested size
def _make_types(n, types):

    try:
        if len(types):
            return types
    except TypeError:
        pass

    if types is None:
        return [m.Particle] * n

    return [types] * n

# hold bond rule info,
#
# *func: function of func(p1, p2) that accepts two particle handles and
#        returns a bond.
#
# *parts: pair of particle ids in current current and other unit cell
#         must be tuple.
#
# *cell_offset: offset vector of other unit cell relative to current
#         unit cell. Must be a tuple
_BondRule = namedtuple('_BondRule', ['func', 'part_ids', 'cell_offset'])

class unitcell(object):
    R""" Define a unit cell.

    Args:
        N (int): Number of particles in the unit cell.
        a1 (list): Lattice vector (3-vector).
        a2 (list): Lattice vector (3-vector).
        a3 (list): Lattice vector (3-vector). Set to [0,0,1] in 2D lattices.
        dimensions (int): Dimensionality of the lattice (2 or 3).
        position (list): List of particle positions.
        type_name (list): List of particle type names.
        mass (list): List of particle masses.
        charge (list): List of particle charges.
        diameter (list): List of particle diameters.
        moment_inertia (list): List of particle moments of inertia.
        orientation (list): List of particle orientations.
        bonds (tuple): a list of tuples, where each tuple that contains a:
                      * potential,
                      * tuple of particle index in current cell, and
                        another unit cell
                      * tuple of the other unit cell's offset, i.e.
                      to bind the 1'th particle in a cell with the 1'th particle
                      in another cell one unit cell offset in the i direction, we
                      would:
                      (pot, (1, 1), (1, 0, 0))

    A unit cell is a box definition (*a1*, *a2*, *a3*, *dimensions*), and particle properties
    for *N* particles. You do not need to specify all particle properties. Any property omitted
    will be initialized to defaults. The :py:class:`create_lattice` initializes the system with
    many copies of a unit cell.

    :py:class:`unitcell` is a completely generic unit cell representation. See other classes in
    the :py:mod:`lattice` module for convenience wrappers for common lattices.

    Example::

        uc = lattice.unitcell(N = 2,
                              a1 = [1,0,0],
                              a2 = [0.2,1.2,0],
                              a3 = [-0.2,0, 1.0],
                              dimensions = 3,
                              position = [[0,0,0], [0.5, 0.5, 0.5]],
                              types = [A, B],
                              orientation = [[0.707, 0, 0, 0.707], [1.0, 0, 0, 0]]);

    Note:
        *a1*, *a2*, *a3* must define a right handed coordinate system.

    """

    def __init__(self,
                 N,
                 a1,
                 a2,
                 a3,
                 dimensions = 3,
                 position = None,
                 types = None,
                 diameter = None,
                 orientation = None,
                 bonds = None):

        self.N = N;
        self.a1 = numpy.asarray(a1, dtype=numpy.float64)
        self.a2 = numpy.asarray(a2, dtype=numpy.float64)
        self.a3 = numpy.asarray(a3, dtype=numpy.float64)
        self.dimensions = dimensions
        self.bonds = bonds

        if position is None:
            self.position = numpy.array([(0,0,0)] * self.N, dtype=numpy.float64);
        else:
            self.position = numpy.asarray(position, dtype=numpy.float64);
            if len(self.position) != N:
                raise ValueError("Particle properties must have length N");

        if types is None:
            self.types = [m.Particle] * self.N
        else:
            self.types = types;
            if len(self.types) != N:
                raise ValueError("Particle properties must have length N");

        if orientation is None:
            self.orientation = numpy.array([(1,0,0,0)] * self.N, dtype=numpy.float64);
        else:
            self.orientation = numpy.asarray(orientation, dtype=numpy.float64);
            if len(self.orientation) != N:
                raise ValueError("Particle properties must have length N");




def sc(a, types=None, bond=None, bond_vector=(True, True, True)):
    R""" Create a simple cubic lattice (3D).

    Args:
        a (float): Lattice constant.
        type_name (str): Particle type name.

    The simple cubic unit cell has 1 particle:

    .. math::
        :nowrap:

        \begin{eqnarray*}
        \vec{r}& =& \left(\begin{array}{ccc} 0 & 0 & 0 \\
                             \end{array}\right)
        \end{eqnarray*}

    And the box matrix:

    .. math::
        :nowrap:

        \begin{eqnarray*}
        \mathbf{h}& =& \left(\begin{array}{ccc} a & 0 & 0 \\
                                                0 & a & 0 \\
                                                0 & 0 & a \\
                             \end{array}\right)
        \end{eqnarray*}
    """

    bonds = None
    if bond:
        bonds = []

        if bond_vector[0]:
            bonds.append(_BondRule(bond, (0,0), (1, 0, 0)))

        if bond_vector[1]:
            bonds.append(_BondRule(bond, (0,0), (0, 1, 0)))

        if bond_vector[2]:
            bonds.append(_BondRule(bond, (0,0), (0, 0, 1)))

    return unitcell(N=1,
                    types=_make_types(1, types),
                    a1=[a,0,0],
                    a2=[0,a,0],
                    a3=[0,0,a],
                    dimensions=3,
                    bonds=bonds);

def bcc(a, types = None):
    R""" Create a body centered cubic lattice (3D).

    Args:
        a (float): Lattice constant.
        type_name (str): Particle type name.

    The body centered cubic unit cell has 2 particles:

    .. math::
        :nowrap:

        \begin{eqnarray*}
        \vec{r}& =& \left(\begin{array}{ccc} 0 & 0 & 0 \\
                                             \frac{a}{2} & \frac{a}{2} & \frac{a}{2} \\
                             \end{array}\right)
        \end{eqnarray*}

    And the box matrix:

    .. math::
        :nowrap:

        \begin{eqnarray*}
        \mathbf{h}& =& \left(\begin{array}{ccc} a & 0 & 0 \\
                                                0 & a & 0 \\
                                                0 & 0 & a \\
                             \end{array}\right)
        \end{eqnarray*}
    """

    return unitcell(N=2,
                    types = _make_types(2, types),
                    position=[[0,0,0],[a/2,a/2,a/2]],
                    a1=[a,0,0],
                    a2=[0,a,0],
                    a3=[0,0,a],
                    dimensions=3);

def fcc(a, types=None):
    R""" Create a face centered cubic lattice (3D).

    Args:
        a (float): Lattice constant.
        type_name (str): Particle type name.

    The face centered cubic unit cell has 4 particles:

    .. math::
        :nowrap:

        \begin{eqnarray*}
        \vec{r}& =& \left(\begin{array}{ccc} 0 & 0 & 0 \\
                                             0 & \frac{a}{2} & \frac{a}{2} \\
                                             \frac{a}{2} & 0 & \frac{a}{2} \\
                                             \frac{a}{2} & \frac{a}{2} & 0\\
                             \end{array}\right)
        \end{eqnarray*}

    And the box matrix:

    .. math::
        :nowrap:

        \begin{eqnarray*}
        \mathbf{h}& =& \left(\begin{array}{ccc} a & 0 & 0 \\
                                                0 & a & 0 \\
                                                0 & 0 & a \\
                             \end{array}\right)
        \end{eqnarray*}
    """

    return unitcell(N=4,
                    types=_make_types(4, types),
                    position=[[0,0,0],[0,a/2,a/2],[a/2,0,a/2],[a/2,a/2,0]],
                    a1=[a,0,0],
                    a2=[0,a,0],
                    a3=[0,0,a],
                    dimensions=3);

def sq(a, types=None):
    R""" Create a square lattice (2D).

    Args:
        a (float): Lattice constant.
        type_name (str): Particle type name.

    The simple square unit cell has 1 particle:

    .. math::
        :nowrap:

        \begin{eqnarray*}
        \vec{r}& =& \left(\begin{array}{ccc} 0 & 0 \\
                             \end{array}\right)
        \end{eqnarray*}

    And the box matrix:

    .. math::
        :nowrap:

        \begin{eqnarray*}
        \mathbf{h}& =& \left(\begin{array}{ccc} a & 0 \\
                                                0 & a \\
                             \end{array}\right)
        \end{eqnarray*}
    """


    return unitcell(N=1,
                    types=_make_types(1, types),
                    a1=[a,0,0],
                    a2=[0,a,0],
                    a3=[0,0,1],
                    dimensions=2);

def hex(a, types=None):
    R""" Create a hexagonal lattice (2D).

    Args:
        a (float): Lattice constant.
        type_name (str): Particle type name.

    :py:class:`hex` creates a hexagonal lattice in a rectangular box.
    It has 2 particles, one at the corner and one at the center of the rectangle.
    This is not the primitive unit cell, but is more convenient to
    work with because of its shape.

    .. math::
        :nowrap:

        \begin{eqnarray*}
        \vec{r}& =& \left(\begin{array}{ccc} 0 & 0 \\
                                             \frac{a}{2} & \sqrt{3} \frac{a}{2} \\
                             \end{array}\right)
        \end{eqnarray*}

    And the box matrix:

    .. math::
        :nowrap:

        \begin{eqnarray*}
        \mathbf{h}& =& \left(\begin{array}{ccc} a & 0 \\
                                                0 & \sqrt{3} a \\
                                                0 & 0 \\
                             \end{array}\right)
        \end{eqnarray*}
    """


    return unitcell(N=2,
                    types=_make_types(2, types),
                    position=[[0,0,0],[a/2,math.sqrt(3)*a/2,0]],
                    a1=[a,0,0],
                    a2=[0,math.sqrt(3)*a,0],
                    a3=[0,0,1],
                    dimensions=2);


def hcp(a, c=None, types=None):
    R""" Create a hexagonal close pack cell

    Args:
        a (float): Lattice constant.
        type_name (str): Particle type name.

    :py:class:`hcp` creates a hexagonal lattice in a rectangular box.
    It has 6 particles, one at the corner and one at the center of the rectangle.
    This is not the primitive unit cell, but is more convenient to
    work with because of its shape.

    .. math::
        :nowrap:

        \begin{eqnarray*}
        \vec{r}& =& \left(\begin{array}{ccc} 0 & 0 \\
                                             \frac{a}{2} & \sqrt{3} \frac{a}{2} \\
                             \end{array}\right)
        \end{eqnarray*}

    And the box matrix:

    .. math::
        :nowrap:

        \begin{eqnarray*}
        \mathbf{h}& =& \left(\begin{array}{ccc} a & 0 \\
                                                0 & \sqrt{3} a \\
                                                0 & 0 \\
                             \end{array}\right)
        \end{eqnarray*}
    """

    if c is None:
        c = a


    return unitcell(N=6,
                    types=_make_types(6, types),
                    position=[[0,0,0],[a/2,math.sqrt(3)*a/2,0]],
                    a1=[a,0,0],
                    a2=[a/2,math.sqrt(3)*a/2,0],
                    a3=[0,0,c],
                    dimensions=3);



def create_lattice(unitcell, n, origin=None):
    R""" Create a lattice.
    Args:
        unitcell (:py:class:`mechanica.lattice.unitcell`):
        The unit cell of the lattice.
        n (list): Number of replicates in each direction.
    :py:func:`create_lattice` take a unit cell and replicates it the requested
    number of times in each direction. The resulting simulation box is commensurate
    with the given unit cell. A generic :py:class:`mechanica.lattice.unitcell`
    may have arbitrary vectors :math:`\vec{a}_1`, :math:`\vec{a}_2`, and
    :math:`\vec{a}_3`. :py:func:`create_lattice` will rotate the unit cell so
    that :math:`\vec{a}_1` points in the :math:`x` direction and
    :math:`\vec{a}_2` is in the :math:`xy` plane so that the lattice may be
    represented as a simulation box. When *n* is a single value, the lattice is
    replicated *n* times in each direction. When *n* is a list, the
    lattice is replicated *n[0]* times in the :math:`\vec{a}_1` direction,
    *n[1]* times in the :math:`\vec{a}_2` direction and *n[2]* times in the
    :math:`\vec{a}_3` direction.

    Examples::
        mechanica.lattice.create_lattice(unitcell=mechanica.lattice.sc(a=1.0),
                                  n=[2,4,2]);
        mechanica.lattice.create_lattice(unitcell=mechanica.lattice.bcc(a=1.0),
                                  n=10);
        mechanica.lattice.create_lattice(unitcell=mechanica.lattice.sq(a=1.2),
                                  n=[100,10]);
        mechanica.lattice.create_lattice(unitcell=mechanica.lattice.hex(a=1.0),
                                  n=[100,58]);
    """

    if origin is None:
        cell_half_size = (unitcell.a1 + unitcell.a2 + unitcell.a3) / 2
        extents = n[0] * unitcell.a1 + n[1] * unitcell.a2 + n[2] * unitcell.a3
        origin = m.Universe.center - extents / 2  + cell_half_size


    lattice = numpy.empty(n, dtype=numpy.object)

    for i in range(n[0]):
        for j in range(n[1]):
            for k in range(n[2]):
                pos = origin + unitcell.a1 * i + unitcell.a2 * j + unitcell.a3 * k;
                parts = [type(pos) for (type,pos) in zip(unitcell.types, unitcell.position + pos)]
                lattice[i,j,k] = parts

    if unitcell.bonds:
        for i in range(n[0]):
            for j in range(n[1]):
                for k in range(n[2]):
                    for bond in unitcell.bonds:
                        ii = (i, j, k) # index of first unit cell, needs to be tuple
                        jj = (ii[0] + bond.cell_offset[0], ii[1] + bond.cell_offset[1], ii[2] + bond.cell_offset[2])
                        # check if next unit cell index is valid
                        if jj[0] >= n[0] or jj[1] >= n[1] or jj[2] >= n[2]:
                            continue

                        #print("ii, jj: ", ii, jj)

                        #print("lattice[(0,0,0)]: ", lattice[(0, 0, 0)])

                        # grap the parts out of the lattice
                        ci = lattice[ii]
                        cj = lattice[jj]

                        #print("ci: ", ci)
                        #print("cj: ", cj)

                        print("bonding: ", ci[bond.part_ids[0]], cj[bond.part_ids[1]])

                        bond.func(ci[bond.part_ids[0]], cj[bond.part_ids[1]])

    return lattice


