/*
 * MeshDampedLangevinPropagator.h
 *
 *  Created on: Aug 3, 2017
 *      Author: andy
 */

#ifndef SRC_MESHDAMPEDLANGEVINPROPAGATOR_H_
#define SRC_MESHDAMPEDLANGEVINPROPAGATOR_H_

#include "MxPropagator.h"

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


    MxModel *model;
    MxMesh *mesh;

};

#endif /* SRC_MESHDAMPEDLANGEVINPROPAGATOR_H_ */
