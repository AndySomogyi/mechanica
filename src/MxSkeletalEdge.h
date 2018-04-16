/*
 * SkeletalEdge.h
 *
 *  Created on: Mar 20, 2018
 *      Author: andy
 */

#ifndef SRC_MXSKELETALEDGE_H_
#define SRC_MXSKELETALEDGE_H_

#include "MxMeshCore.h"

struct MxSkeletalEdge
{

    MxSkeletalEdge();
    ~MxSkeletalEdge();

    MxObject *next;

    MxObject *prev;
};

#endif /* SRC_MXSKELETALEDGE_H_ */
