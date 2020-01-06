/*
 * MxParticles.h
 *
 *  Created on: Feb 25, 2017
 *      Author: andy
 */

#ifndef SRC_MXPARTICLES_H_
#define SRC_MXPARTICLES_H_

#include "mechanica_private.h"
#include "glm/glm.hpp"


/**
 * Represents a collection of simple particles.
 *
 * Simple particles are spherical, and are soft-spheres. They have mass (inverse mass),
 * and might have orientation vector.
 */

struct MxParticles : CObject {
	glm::vec3 *positions;
	glm::vec3 *velocties;
	glm::ivec2 *texCoords;
};

#endif /* SRC_MXPARTICLES_H_ */
