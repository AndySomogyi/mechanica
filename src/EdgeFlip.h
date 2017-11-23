/*
 * EdgeFlip.h
 *
 *  Created on: Nov 23, 2017
 *      Author: andy
 */

#ifndef SRC_EDGEFLIP_H_
#define SRC_EDGEFLIP_H_

#include "MeshOperations.h"

struct EdgeFlip : MeshOperation {
    EdgeFlip(MeshPtr mesh, const Edge& endge);

    static bool applicable(const Edge& e);
};

#endif /* SRC_EDGEFLIP_H_ */
