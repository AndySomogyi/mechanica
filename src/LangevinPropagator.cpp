/*
 * MeshDampedLangevinPropagator.cpp
 *
 *  Created on: Aug 3, 2017
 *      Author: andy
 */

#include <LangevinPropagator.h>
#include <MxModel.h>
#include "stochastic_rk.h"

LangevinPropagator::LangevinPropagator(MxModel* m) :
    model{m}, mesh{m->mesh} {
}

HRESULT LangevinPropagator::step(MxReal dt) {

    TrianglePtr *tris = mesh->triangles.data();

    HRESULT result;

    if((result = model->calcForce(
            tris, (uint32_t)mesh->triangles.size())) != S_OK) {
        return result;
    }

    if((result = eulerStep(dt)) != S_OK) {
        return result;
    }

    return mesh->positionsChanged();
}

HRESULT LangevinPropagator::eulerStep(MxReal dt) {

    for(int i = 0; i < mesh->vertices.size(); ++i) {
        VertexPtr v = mesh->vertices[i];

        assert(v->mass > 0 && v->area > 0);

        float len = v->force.length();
        float tmp = len / v->mass;
        int tri = v->triangles.size();
        float a = v->force.length() / v->mass;

        v->position = v->position + dt * v->force / v->mass;

    }

    return S_OK;
}
