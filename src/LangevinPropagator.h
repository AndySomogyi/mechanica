/*
 * MeshDampedLangevinPropagator.h
 *
 *  Created on: Aug 3, 2017
 *      Author: andy
 */

#ifndef SRC_MESHDAMPEDLANGEVINPROPAGATOR_H_
#define SRC_MESHDAMPEDLANGEVINPROPAGATOR_H_

#include "MxPropagator.h"
#include "ForceAccumulator.h"

/**
 * Damped Langevin propagator,
 *
 * Calculates time evolution via the over-damped Langevin equation,
 *
 * m dx/dt = F(x)/ \gamma + \eta(t)
 */
class LangevinPropagator {



public:

    LangevinPropagator(MxMesh *msh);

    HRESULT step(MxReal dt);

private:


    HRESULT eulerStep(MxReal dt);


    MxMesh *mesh;



    ForceAccumulator forceAccumulator;


};

#endif /* SRC_MESHDAMPEDLANGEVINPROPAGATOR_H_ */
