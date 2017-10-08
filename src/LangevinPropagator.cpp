/*
 * MeshDampedLangevinPropagator.cpp
 *
 *  Created on: Aug 3, 2017
 *      Author: andy
 */

#include <LangevinPropagator.h>

LangevinPropagator::LangevinPropagator(MxMesh* msh) :
    mesh{msh} {
}

HRESULT LangevinPropagator::step(MxReal dt) {

    TrianglePtr *tris = mesh->triangles.data();

    HRESULT result;

    if((result = forceAccumulator.calculateForce(
            tris, (uint32_t)mesh->triangles.size())) != S_OK) {
        return result;
    }

    if((result = eulerStep(dt)) != S_OK) {
        return result;
    }

    return mesh->positionsChanged();
}

HRESULT LangevinPropagator::eulerStep(MxReal dt) {
    return S_OK;
}
