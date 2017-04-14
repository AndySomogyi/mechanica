/*
 * mx_lattice.h
 *
 *  Created on: Apr 8, 2017
 *      Author: andy
 */

#ifndef INCLUDE_MX_LATTICE_H_
#define INCLUDE_MX_LATTICE_H_

#include "mx_list.h"


/**
 * The MxLattice structure represents a regular lattice structure that occupies a region of
 * space. The lattice itself does not have to be convex or regular shapped, it is a
 * collection of UnitCells ordered into a regular lattice structure.
 *
 * Lattices act as fixed boundary conditions for other partices. Lattices move in space
 * accordign to local update rules. They move by creating new unit cells on a leading edge,
 * and destroying unit cells on a trailing edge.
 *
 * Each Lattice object presently aligned to an imaginary global lattice, however, only unit
 * cell storage space is only allocated for small regions.
 *
 * A MxLattice may contain several different UnitCell derived types, the UnitCells in a
 * MxLattice may change type from one UnitCell derived type to another. The restriction
 * is that a MxLattice can only contain appropriate UnitCell types, i.e. a cubic Lattice
 * can only contain cubic UnitCell derived types.
 *
 *
 * A Lattice can track collections or groupings of unit cells, called 'Clusters'. These are
 * analagous to a magnetic domain or cluster of like oriented spins in an Ising model. Each
 * cluster consits of a set of one or more unit cells of the same type.
 */
MxAPI_STRUCT(MxLattice);
MxAPI_DATA(MxType*) MxLattice_Type;

MxAPI_STRUCT(MxCubicLattice);
MxAPI_DATA(MxType*) MxCubicLattice_Type;

MxAPI_STRUCT(MxVoxel);
MxAPI_DATA(MxType*) MxVoxel_Type;

MxAPI_STRUCT(MxCubicVoxel);
MxAPI_DATA(MxType*) MxCubicVoxel_Type;

MxAPI_STRUCT(MxLatticeCluster);
MxAPI_DATA(MxType*) MxLatticeCluster_Type;

MxAPI_STRUCT(MxCubicLatticeCluster);
MxAPI_DATA(MxType*) MxCubicLatticeCluster_Type;




/**
 * Get a borrowed reference to the cluster list.
 */
MxAPI_FUNC(MxList*) MxLattice_Clusters();

/**
 * Borrowed referenct to list of voxels.
 */
MxAPI_FUNC(MxList*) MxLattice_Voxels();










#endif /* INCLUDE_MX_LATTICE_H_ */
