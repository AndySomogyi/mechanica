/*
 * MxParticle.h
 *
 *  Created on: Apr 7, 2017
 *      Author: andy
 */

#ifndef SRC_MXPARTICLE_H_
#define SRC_MXPARTICLE_H_

#include "mechanica_private.h"

struct MxParticle : CObject {
};


/**
 * Init and add to python module
 */
void MxParticle_init(PyObject *m);


#endif /* SRC_MXPARTICLE_H_ */
