/*
 * MxSkeletalVertex.h
 *
 *  Created on: Mar 20, 2018
 *      Author: andy
 */

#ifndef SRC_MXSKELETALVERTEX_H_
#define SRC_MXSKELETALVERTEX_H_

#include "MxMeshCore.h"

#define SKELETAL_VERTEX_MAX_EDGES 4

struct MxSkeletalEdge;


struct MxSkeletalVertex : MxVertex
{
public:
    MxSkeletalVertex();
    ~MxSkeletalVertex();


    unsigned edgeCount;

    /**
     * Fixed number of edges
     */
    MxSkeletalEdge *edges[SKELETAL_VERTEX_MAX_EDGES];
};

#endif /* SRC_SKELETALVERTEX_H_ */
