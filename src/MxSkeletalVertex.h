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

MxAPI_DATA(struct MxType*) MxSkeletalVertex_Type;


struct MxSkeletalVertex : MxVertex
{
public:
    MxSkeletalVertex();
    ~MxSkeletalVertex();

    MxSkeletalVertex(float mass, float area, const Magnum::Vector3 &pos);

    HRESULT init(float mass, float area, const Magnum::Vector3 &pos);


    unsigned edgeCount = 0;

    /**
     * Fixed number of edges
     */
    MxSkeletalEdge *edges[SKELETAL_VERTEX_MAX_EDGES];

    static bool classof(const MxObject *o) {
        return o->ob_type == MxSkeletalVertex_Type;
    }
};

#endif /* SRC_SKELETALVERTEX_H_ */
