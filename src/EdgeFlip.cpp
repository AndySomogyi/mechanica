/*
 * EdgeFlip.cpp
 *
 *  Created on: Nov 23, 2017
 *      Author: andy
 */

#include <EdgeFlip.h>


EdgeFlip::EdgeFlip(MeshPtr mesh, const Edge& _edge) : MeshOperation{mesh} {
}

bool EdgeFlip::applicable(const Edge& _e) {
    return false;
}
