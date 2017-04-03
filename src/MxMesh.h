/*
 * MxMesh.h
 *
 *  Created on: Jan 31, 2017
 *      Author: andy
 */

#ifndef _INCLUDE_MXMESH_H_
#define _INCLUDE_MXMESH_H_

#include "mechanica_private.h"
#include "glm/glm.hpp"

/**
 * Internal implementation of MxObject
 */
struct MxMesh : MxObject {
	glm::vec3 *positions;
	glm::vec3 *normals;
	glm::vec3 *velocties;
	glm::ivec2 *texCoords;

};

#endif /* _INCLUDE_MXMESH_H_ */
