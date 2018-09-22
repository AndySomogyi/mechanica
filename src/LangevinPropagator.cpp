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



LangevinPropagator::LangevinPropagator() {
}

HRESULT LangevinPropagator::setModel(MxModel *m) {
    this->model = m;
    this->mesh = m->mesh;
    m->propagator = this;
    
    return structureChanged();
}

HRESULT LangevinPropagator::updateConstraints() {
    for(ConstraintItems& ci : constraints) {
        updateConstraint(ci);
    }
    return S_OK;
}

HRESULT LangevinPropagator::step(MxReal dt) {

    HRESULT result = S_OK;

    resize();


    for(int i = 0; i < 10; ++i) {
        if((result = rungeKuttaStep(dt/10)) != S_OK) {
            return result;
        }
    }


    if((timeSteps % 20) == 0) {
        result = mesh->applyMeshOperations();
    }

#ifdef NEW_CONSTRAINTS

    applyConstraints();

#else


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



#endif



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
    if(!mesh) {
        return;
    }
    
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

HRESULT LangevinPropagator::getAccelerations(float time, uint32_t len,
        const Vector3* pos, Vector3* acc)
{
}

HRESULT LangevinPropagator::getPositions(float time, uint32_t len, Vector3* pos)
{
}

HRESULT LangevinPropagator::applyConstraints()
{


    float sumError = 0;
    int iter = 0;

    do {

        for(ConstraintItems &ci : constraints) {
            MxObject **data = ci.args.data();
            ci.constraint->project(data, ci.args.size());
        }
        
        iter += 1;

    } while(iter < 2);
    return S_OK;
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

HRESULT LangevinPropagator::structureChanged()
{
    if(!model) {
        return S_OK;
    }
    
    mesh = model->mesh;
    
    resize();

    return updateConstraints();
}

HRESULT LangevinPropagator::bindConstraint(IConstraint* constraint,
        MxObject* obj)
{
    MxType *type = dyn_cast<MxType>(obj);
    if(type) {
        return bindTypeConstraint(constraint, type);
    }
    return E_NOTIMPL;
}

HRESULT LangevinPropagator::bindForce(IForce* force, MxObject* obj)
{
    return E_NOTIMPL;
}

HRESULT LangevinPropagator::updateConstraint(ConstraintItems& ci) {
    HRESULT result = E_FAIL;
    
    if(!mesh) {
        return S_OK;
    }
    
    ci.args.clear();
    
    if (MxType_IsSubtype(ci.type, MxCell::type())) {
        for(CellPtr cell : mesh->cells) {
            if(cell->isRoot()) {
                continue;
            }
            
            if(MxType_IsSubtype(cell->ob_type, ci.type)) {
                ci.args.push_back(cell);
                result = S_OK;
            }
        }
    }
    
    if (MxType_IsSubtype(ci.type, MxPolygon::type())) {
        for(PolygonPtr poly : mesh->polygons) {
            if(MxType_IsSubtype(poly->ob_type, ci.type)) {
                ci.args.push_back(poly);
                result = S_OK;
            }
        }
    }
    
    std::cout << "constraint for " << ci.type->tp_name << " , args size: " << ci.args.size() << std::endl;
    
    return result;
}

HRESULT LangevinPropagator::bindTypeConstraint(IConstraint* constraint,
        MxType* type)
{
    ConstraintItems& ci = getConstraintItem(constraint);
    ci.type = type;
    return updateConstraint(ci);
}

HRESULT LangevinPropagator::bindTypeForce(IForce* force, MxType* obj)
{
    return E_NOTIMPL;
}

LangevinPropagator::ConstraintItems& LangevinPropagator::getConstraintItem(IConstraint* cons)
{
    auto it = std::find_if(
            constraints.begin(), constraints.end(),
            [cons](const ConstraintItems& x) { return x.constraint == cons;});
    if(it != constraints.end()) {
        return *it;
    }
    else {
        constraints.push_back(ConstraintItems{cons});
        return constraints.back();
    }
}

HRESULT MxBind_PropagatorModel(LangevinPropagator* propagator, MxModel* model)
{
    model->propagator = propagator;
    return propagator->setModel(model);
}
