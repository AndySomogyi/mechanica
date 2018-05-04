/*
 * SkeletalEdgeSplit.h
 *
 *  Created on: Apr 23, 2018
 *      Author: andy
 */

#ifndef SRC_SKELETALEDGESPLIT_H_
#define SRC_SKELETALEDGESPLIT_H_

#include "MxMesh.h"
#include "MxSkeletalEdge.h"
#include "MxSkeletalVertex.h"

class SkeletalEdgeSplit
{
};

/**
 * Split a skeletal edge along its length at a given point.
 *
 * A skeletal edge is attached to two skeletal vertices on each end. A skeletal edge is incident
 * to exactly three cells (for now), and exactly three triangles.
 *
 * destroys edge, and creates two new edges, e1 and e2.
 */
HRESULT skeletalEdgeSpit(MxMesh *mesh, MxSkeletalEdge *edge, const Vector3& midpt,
        MxSkeletalEdge **e1, MxSkeletalEdge **e2, MxSkeletalVertex **vert);

#endif /* SRC_SKELETALEDGESPLIT_H_ */
