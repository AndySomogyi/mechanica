/*
 * SkeletalEdge.h
 *
 *  Created on: Mar 20, 2018
 *      Author: andy
 */

#ifndef SRC_MXSKELETALEDGE_H_
#define SRC_MXSKELETALEDGE_H_

#include "MxMeshCore.h"


#define SKELETAL_EDGE_MAX_TRIANGLES 3

/**
 * A skeletal edge is at the boundary of three or more cells.
 */
struct MxSkeletalEdge : MxObject
{

    MxSkeletalEdge();
    ~MxSkeletalEdge();

    MxObject *next;

    MxObject *prev;

    MxTriangle *triangles[SKELETAL_EDGE_MAX_TRIANGLES];
};

#endif /* SRC_MXSKELETALEDGE_H_ */
