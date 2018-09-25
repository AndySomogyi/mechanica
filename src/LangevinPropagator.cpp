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


template<typename T>
HRESULT LangevinPropagator::updateItem(T &item) {
    HRESULT result = E_FAIL;

    if(!mesh) {
        return S_OK;
    }

    item.args.clear();

    if (MxType_IsSubtype(item.type, MxCell::type())) {
        for(CellPtr cell : mesh->cells) {
            if(cell->isRoot()) {
                continue;
            }

            if(MxType_IsSubtype(cell->ob_type, item.type)) {
                item.args.push_back(cell);
                result = S_OK;
            }
        }
    }

    if (MxType_IsSubtype(item.type, MxPolygon::type())) {
        for(PolygonPtr poly : mesh->polygons) {
            if(MxType_IsSubtype(poly->ob_type, item.type)) {
                item.args.push_back(poly);
                result = S_OK;
            }
        }
    }

    std::cout << "items for " << item.type->tp_name << " , args size: " << item.args.size() << std::endl;

    return result;
}


template<typename T>
HRESULT LangevinPropagator::updateItems(std::vector<T> &items)
{
    for(T& i : items) {
        updateItem(i);
    }
    return S_OK;
}

template<typename T, typename KeyType>
T& LangevinPropagator::getItem(std::vector<T>& items, KeyType* key)
{
    auto it = std::find_if(
            items.begin(), items.end(),
            [key](const T& x) { return x.thing == key;});
    if(it != items.end()) {
        return *it;
    }
    else {
        items.push_back(T{key});
        return items.back();
    }
}

template<typename T, typename KeyType>
HRESULT LangevinPropagator::bindTypeItem(std::vector<T>& items,
        KeyType* key, MxType* type)
{
    T& ci = getItem(items, key);
    ci.type = type;
    return updateItem(ci);
}

HRESULT LangevinPropagator::setModel(MxModel *m) {
    this->model = m;
    this->mesh = m->mesh;
    m->propagator = this;
    
    mesh->addObjectDeleteListener(objectDeleteListener, this);

    return structureChanged();
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

    applyConstraints();

    timeSteps += 1;

    return result;
}

HRESULT LangevinPropagator::eulerStep(MxReal dt) {

    getPositions(dt, size, positions);

    getAccelerations(dt, size, positions, accel);

    for(int i = 0; i < size; ++i) {
        positions[i] = positions[i] + dt * accel[i];
    }

    setPositions(dt, size, positions);

    return S_OK;
}

HRESULT LangevinPropagator::rungeKuttaStep(MxReal dt)
{
    getAccelerations(dt, size, nullptr, k1);

    getPositions(dt, size, posInit);

    for(int i = 0; i < size; ++i) {
        positions[i] = posInit[i] + dt * k1[i] / 2.0 ;
    }

    getAccelerations(dt, size, positions, k2);

    for(int i = 0; i < size; ++i) {
        positions[i] = posInit[i] + dt * k2[i] / 2.0 ;
    }

    getAccelerations(dt, size, positions, k3);

    for(int i = 0; i < size; ++i) {
        positions[i] = posInit[i] + dt * k3[i];
    }

    getAccelerations(dt, size, positions, k4);

    for(int i = 0; i < size; ++i) {
        positions[i] = posInit[i] + dt / 6. * (k1[i] + 2. * k2[i] + 2. * k3[i] + k4[i]);
    }

    setPositions(dt, size, positions);

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
    HRESULT result;

    if(len != mesh->vertices.size()) {
        return E_FAIL;
    }

    if(pos) {
        if(!SUCCEEDED(result = mesh->setPositions(len, pos))) {
            return result;
        }
    }

    VERIFY(applyForces());

    for(int i = 0; i < mesh->vertices.size(); ++i) {
        VertexPtr v = mesh->vertices[i];

        acc[i] = v->force;
    }

    return S_OK;
}

HRESULT LangevinPropagator::getPositions(float time, uint32_t len, Vector3* pos)
{
    for(int i = 0; i < len; ++i) {
        pos[i] = mesh->vertices[i]->position;
    }
    return S_OK;
}

HRESULT LangevinPropagator::applyConstraints()
{
    float sumError = 0;
    int iter = 0;

    do {

        for(ConstraintItems &ci : constraints) {
            MxObject **data = ci.args.data();
            ci.thing->project(data, ci.args.size());
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

    VERIFY(updateItems(forces));

    VERIFY(updateItems(constraints));

    return S_OK;
}


HRESULT LangevinPropagator::bindForce(IForce* force, MxObject* obj)
{
    MxType *type = dyn_cast<MxType>(obj);
    if(type) {
        return bindTypeItem(forces, force, type);
    }
    return E_NOTIMPL;
}

HRESULT LangevinPropagator::bindConstraint(IConstraint* constraint,
        MxObject* obj)
{
    MxType *type = dyn_cast<MxType>(obj);
    if(type) {
        return bindTypeItem(constraints, constraint, type);
    }
    return E_NOTIMPL;
}

HRESULT MxBind_PropagatorModel(LangevinPropagator* propagator, MxModel* model)
{
    model->propagator = propagator;
    return propagator->setModel(model);
}

HRESULT LangevinPropagator::objectDeleteListener(MxObject* pThis,
        const MxObject* obj, uint32_t what)
{
}

HRESULT LangevinPropagator::unbindConstraint(IConstraint* constraint)
{
}

HRESULT LangevinPropagator::unbindForce(IForce* force)
{
}

HRESULT LangevinPropagator::setPositions(float time, uint32_t len, const Vector3* pos)
{
    return mesh->setPositions(len, pos);
}

HRESULT LangevinPropagator::applyForces()
{
    for(ForceItems &f : forces) {
        MxObject **data = f.args.data();
        f.thing->applyForce(0, data, f.args.size());
    }

    return S_OK;
}
