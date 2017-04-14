/*
 * MxLattice.h
 *
 *  Created on: Apr 8, 2017
 *      Author: andy
 */

#ifndef SRC_MXLATTICE_H_
#define SRC_MXLATTICE_H_

#include "mechanica_private.h"

struct MxLattice : MxObject {
};


struct MxCubicLattice : MxLattice {

};

/**
 * The internal cubic unit cell type, if this is accessed externally, each one
 * gets wrapped in a MxCubicUnitCell type.
 */
struct CubicVoxel {

};

struct MxCubicVoxel : MxObject {
	/**
	 * each unitcell is owned by the lattice.
	 */
	CubicVoxel *unitCell;
};



#endif /* SRC_MXLATTICE_H_ */
