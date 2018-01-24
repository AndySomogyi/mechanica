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
#include <cstdlib>
#include <cstring>

LangevinPropagator::LangevinPropagator(MxModel* m) :
    model{m}, mesh{m->mesh} {
    resize();
}

HRESULT LangevinPropagator::step(MxReal dt) {

    HRESULT result = S_OK;

    resize();

    if((result = rungeKuttaStep(dt)) != S_OK) {
        return result;
    }

    if((timeSteps % 15) == 0) {
        result = mesh->applyMeshOperations();
    }

    timeSteps += 1;

    return result;
}

HRESULT LangevinPropagator::eulerStep(MxReal dt) {

    model->getPositions(dt, size, positions);

    model->getAccelerations(dt, size, positions, accel);

    for(int i = 0; i < size; ++i) {
        positions[i] = positions[i] + dt * accel[i];
    }

    model->setPositions(dt, size, positions);

    return S_OK;
}

HRESULT LangevinPropagator::rungeKuttaStep(MxReal dt)
{
    model->getAccelerations(dt, size, nullptr, k1);

    model->getPositions(dt, size, posInit);

    for(int i = 0; i < size; ++i) {
        positions[i] = posInit[i] + dt * k1[i] / 2.0 ;
    }

    model->getAccelerations(dt, size, positions, k2);

    for(int i = 0; i < size; ++i) {
        positions[i] = posInit[i] + dt * k2[i] / 2.0 ;
    }

    model->getAccelerations(dt, size, positions, k3);

    for(int i = 0; i < size; ++i) {
        positions[i] = posInit[i] + dt * k3[i];
    }

    model->getAccelerations(dt, size, positions, k4);

    for(int i = 0; i < size; ++i) {
        positions[i] = posInit[i] + dt / 6. * (k1[i] + 2. * k2[i] + 2. * k3[i] + k4[i]);
    }

    model->setPositions(dt, size, positions);

    return S_OK;
}

void LangevinPropagator::resize()
{
    if(size != mesh->vertices.size()) {
        size = mesh->vertices.size();
        positions = (Vector3*)std::realloc(positions, size * sizeof(Vector3));
        accel = (Vector3*)std::realloc(accel, size * sizeof(Vector3));
        masses = (float*)std::realloc(masses, size * sizeof(float));

        posInit = (Vector3*)std::realloc(posInit, size * sizeof(Vector3));
        k1 = (Vector3*)std::realloc(k1, size * sizeof(Vector3));
        k2 = (Vector3*)std::realloc(k2, size * sizeof(Vector3));
        k3 = (Vector3*)std::realloc(k3, size * sizeof(Vector3));
        k4 = (Vector3*)std::realloc(k4, size * sizeof(Vector3));
    }
}
