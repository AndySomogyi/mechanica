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

MxAPI_DATA(struct MxType*) MxSkeletalEdge_Type;

struct MxSkeletalVertex;

/**
 * A skeletal edge is at the boundary of three or more cells.
 */
struct MxSkeletalEdge : MxObject
{

    MxSkeletalEdge();
    ~MxSkeletalEdge();

    /**
     * The next and prev pointers are a skeletal vertex.
     */

    MxSkeletalVertex *vertices[2];


    MxTriangle *triangles[SKELETAL_EDGE_MAX_TRIANGLES] = {nullptr};

    static bool classof(const MxObject *o) {
        return o->ob_type == MxSkeletalEdge_Type;
    }
};

#endif /* SRC_MXSKELETALEDGE_H_ */
