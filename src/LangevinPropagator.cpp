/*
 * MeshDampedLangevinPropagator.cpp
 *
 *  Created on: Aug 3, 2017
 *      Author: andy
 */

#include <LangevinPropagator.h>
#include <MxModel.h>
#include "stochastic_rk.h"
#include <iostream>

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

        Magnum::Vector3 force;

        for(CTrianglePtr tri : v->triangles()) {
            for(int j = 0; j < 3; ++j) {
                if(tri->vertices[j] == v) {
                    force += tri->force[j];
                }
            }
        }
        
        v->position = v->position + dt * force / v->mass;
    }

    return S_OK;
}
