/*
 * RadialEdgeFlip.cpp
 *
 *  Created on: Nov 30, 2017
 *      Author: andy
 */

#include "RadialEdgeFlip.h"


RadialEdgeFlip::RadialEdgeFlip(MeshPtr msh, VertexPtr vert) :
    MeshOperation{msh}, vertex{vert} {
}


bool RadialEdgeFlip::applicable(const Edge& e) {
}

/**
 * Apply this operation
 */
HRESULT RadialEdgeFlip::apply() {
    return E_NOTIMPL;
}

/**
 * lower, more negative energy operations are queued at a higher priority.
 */
float RadialEdgeFlip::energy() const {
    return false;
}

/**
 * does this operation depend on this triangle?
 */
bool RadialEdgeFlip::depends(CTrianglePtr) const {
    return false;
}

/**
 * does this operation depend on this vertex?
 */
bool RadialEdgeFlip::depends(CVertexPtr) const {
    return false;
}

bool RadialEdgeFlip::equals(const Edge& e) const {
    return false;
}

void RadialEdgeFlip::mark() const {

}

