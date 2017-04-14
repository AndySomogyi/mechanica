/*
 * MxModel.h
 *
 *  Created on: Apr 2, 2017
 *      Author: andy
 */

#ifndef SRC_MXMODEL_H_
#define SRC_MXMODEL_H_

#include "mechanica_private.h"

struct MxModel : MxObject {

	/**
	 * Lattice (optional)
	 */
	MxLattice *lattice;

};

void MxModel_init(PyObject *m);

#endif /* SRC_MXMODEL_H_ */
