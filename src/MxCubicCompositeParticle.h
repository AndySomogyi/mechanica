/*
 * MxCubicCompositeParticle.h
 *
 *  Created on: Apr 7, 2017
 *      Author: andy
 */

#ifndef SRC_MXCUBICCOMPOSITEPARTICLE_H_
#define SRC_MXCUBICCOMPOSITEPARTICLE_H_

#include "mechanica_private.h"
#include "MxParticle.h"

/**
 * The MxCubicCompositeParticle represents a region of BCC particles. It effectivly
 * defines a local lattice, and acts as a boundary condition for the regular particles.
 *
 * MxCubicCompositeParticle do not move via conventional forces. Rather, they have local
 * rules and move by moving thier 'voxels'.
 */
struct MxCubicCompositeParticle : MxParticle {
};

#endif /* SRC_MXCUBICCOMPOSITEPARTICLE_H_ */
