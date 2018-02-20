/*
 * MeshDampedLangevinPropagator.h
 *
 *  Created on: Aug 3, 2017
 *      Author: andy
 */

#ifndef SRC_MESHDAMPEDLANGEVINPROPAGATOR_H_
#define SRC_MESHDAMPEDLANGEVINPROPAGATOR_H_

#include "MxPropagator.h"
#include "Magnum/Magnum.h"

struct MxModel;



/**
 * Damped Langevin propagator,
 *
 * Calculates time evolution via the over-damped Langevin equation,
 *
 * m dx/dt = F(x)/ \gamma + \eta(t)
 */
class LangevinPropagator {



public:

    LangevinPropagator(MxModel *model);

    HRESULT step(MxReal dt);

private:


    HRESULT eulerStep(MxReal dt);

    HRESULT rungeKuttaStep(MxReal dt);


    MxModel *model;
    MxMesh *mesh;

    size_t size = 0;
    Magnum::Vector3 *positions = nullptr;

    Magnum::Vector3 *posInit = nullptr;

    Magnum::Vector3 *accel = nullptr;

    Magnum::Vector3 *k1 = nullptr;
    Magnum::Vector3 *k2 = nullptr;
    Magnum::Vector3 *k3 = nullptr;
    Magnum::Vector3 *k4 = nullptr;

    float *masses = nullptr;

    void resize();

    size_t timeSteps = 0;

    uint32_t stateVectorSize = 0;
    float *stateVectorInit = nullptr;
    float *stateVector = nullptr;

    float *stateVectorK1 = nullptr;
    float *stateVectorK2 = nullptr;
    float *stateVectorK3 = nullptr;
    float *stateVectorK4 = nullptr;

    HRESULT stateVectorStep(MxReal dt);

};

#endif /* SRC_MESHDAMPEDLANGEVINPROPAGATOR_H_ */
