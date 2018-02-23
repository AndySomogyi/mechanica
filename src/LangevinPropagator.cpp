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

    //for(int i = 0; i < 100; ++i) {
    //    stateVectorStep(dt);
    //}

    for(int i = 0; i < 10; ++i) {
    if((result = rungeKuttaStep(dt/10)) != S_OK) {
        return result;
    }
    }

    if((timeSteps % 20) == 0) {

        result = mesh->applyMeshOperations();

    }

    float sumError = 0;
    int iter = 0;

    do {
        for(int i=1; i < mesh->cells.size(); ++i) {
            CellPtr cell = mesh->cells[i];
            float init = cell->volumeConstraint();
            cell->projectVolumeConstraint();
            mesh->setPositions(0, 0);
            float final = cell->volumeConstraint();



            std::cout << "cell " << cell->id << " volume constraint before/after: " <<
                    init << "/" << final << std::endl;
        }

        sumError = 0;
        iter += 1;
        for(int i=1; i < mesh->cells.size(); ++i) {
            CellPtr cell = mesh->cells[i];
            float error = cell->volumeConstraint();
            sumError += error * error;
        }

        std::cout << "constraint iter / sum sqr error: " << iter << "/" << sumError << std::endl;

    } while(iter < 2);

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

    uint32_t ssCount = 0;
    model->getStateVector(nullptr, &ssCount);
    if(stateVectorSize != ssCount) {
        stateVectorSize = ssCount;
        stateVector = (float*)std::realloc(stateVector, stateVectorSize * sizeof(float));
        stateVectorInit = (float*)std::realloc(stateVectorInit, stateVectorSize * sizeof(float));
        stateVectorK1 = (float*)std::realloc(stateVectorK1, stateVectorSize * sizeof(float));
        stateVectorK2 = (float*)std::realloc(stateVectorK2, stateVectorSize * sizeof(float));
        stateVectorK3 = (float*)std::realloc(stateVectorK3, stateVectorSize * sizeof(float));
        stateVectorK4 = (float*)std::realloc(stateVectorK4, stateVectorSize * sizeof(float));
    }
}

HRESULT LangevinPropagator::stateVectorStep(MxReal dt)
{
    uint32_t count;
    model->getStateVector(stateVectorInit, &count);
    model->getStateVectorRate(dt, stateVector, stateVectorK1);

    for(int i = 0; i < count; ++i) {
        stateVector[i] = stateVectorInit[i] + dt * stateVectorK1[i] / 2.0 ;
    }

    model->getStateVectorRate(dt, stateVector, stateVectorK2);

    for(int i = 0; i < count; ++i) {
        stateVector[i] = stateVectorInit[i] + dt * stateVectorK2[i] / 2.0 ;
    }

    model->getStateVectorRate(dt, stateVector, stateVectorK3);

    for(int i = 0; i < count; ++i) {
        stateVector[i] = stateVectorInit[i] + dt * stateVectorK3[i];
    }

    model->getStateVectorRate(dt, stateVector, stateVectorK4);

    for(int i = 0; i < count; ++i) {
        stateVector[i] = stateVectorInit[i] + dt / 6. * (
                stateVectorK1[i] +
                2. * stateVectorK2[i] +
                2. * stateVectorK3[i] +
                stateVectorK4[i]
            );
    }

    model->setStateVector(stateVector);

    return S_OK;
}
