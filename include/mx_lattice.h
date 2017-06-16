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
 *
 * The MxLattice itself is just an interface that enables grid like access to values. The
 * MxLattice itself may or may not actually allocate a real lattice, but this is of no concern
 * to callers. All users need be conserned about is that they can read or write attributes
 * at (i,j,k) indexed locations.
 *
 * The MxLattice may be used as a core data structure for Potts or Cellular Automota
 * type dynamics.
 */
MxAPI_STRUCT(MxLattice);
MxAPI_DATA(MxType*) MxLattice_Type;

MxAPI_STRUCT(MxCubicLattice);
MxAPI_DATA(MxType*) MxCubicLattice_Type;

MxAPI_STRUCT(MxVoxel);
MxAPI_DATA(MxType*) MxVoxel_Type;

MxAPI_STRUCT(MxCubicVoxel);
MxAPI_DATA(MxType*) MxCubicVoxel_Type;


/**
 * The MxLatticeCluster represents a collection of fully connected voxels.
 */
MxAPI_STRUCT(MxLatticeCluster);
MxAPI_DATA(MxType*) MxLatticeCluster_Type;

MxAPI_STRUCT(MxCubicLatticeCluster);
MxAPI_DATA(MxType*) MxCubicLatticeCluster_Type;




/**
 * Get a borrowed reference to the cluster list.
 */
MxAPI_FUNC(MxList*) MxLattice_Clusters(MxLattice *p);




/**
 * Set the voxel at a particular location to belong to a cluster type.
 *
 * This causes the set of clusters to be re-ordered. If a newly set voxel is not
 * near an existing cluster of the corresponding type, then a new cluster is
 * created. If a voxel is set near an existing cluster, then that cluster is enlarged,
 * and the specified voxel will then belong to that cluster. If a cluster holds
 * only a single voxel, and the voxel changes type, then the cluster is deleted.
 */
//MxAPI_FUNC(HRESULT) MxLattice_SetVoxelClusterType(MxLattice *lattice,
//		uint x, uint y, uint z,  MxType *type);










#endif /* INCLUDE_MX_LATTICE_H_ */
