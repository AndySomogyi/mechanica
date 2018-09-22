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
#include <Magnum/Math/Vector3.h>
#include "MxConstraints.h"
#include "MxForces.h"

struct MxModel;



/**
 * Damped Langevin propagator,
 *
 * Calculates time evolution via the over-damped Langevin equation,
 *
 * m dx/dt = F(x)/ \gamma + \eta(t)
 */
class LangevinPropagator {

    typedef Magnum::Vector3 Vector3;



public:

    LangevinPropagator(MxModel *model);

    HRESULT step(MxReal dt);

    /**
     * Inform the propagator that the model structure changed.
     */
    HRESULT structureChanged();


    HRESULT bindConstraint(IConstraint *constraint, MxObject *obj);

    HRESULT bindForce(IForce *force, MxObject *obj);

private:

    struct ConstraintItems {
        IConstraint *constraint;
        std::vector<MxObject*> args;
    };

    struct ForceItems {
        IForce *force;
        std::vector<MxObject*> args;
    };


    HRESULT eulerStep(MxReal dt);

    HRESULT rungeKuttaStep(MxReal dt);


    HRESULT getAccelerations(float time, uint32_t len, const Vector3 *pos, Vector3 *acc);

    //HRESULT getMasses(float time, uint32_t len, float *masses);

    HRESULT getPositions(float time, uint32_t len, Vector3 *pos);

    HRESULT applyConstraints();


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


    /**
     * Keep track of constrained objects, most objects aren't constrained.
     */
    std::vector<ConstraintItems> constraints;
    std::vector<ForceItems> forces;

    HRESULT bindTypeConstraint(IConstraint *constraint, MxType *type);

    HRESULT bindTypeForce(IForce *force, MxType *obj);

};

#endif /* SRC_MESHDAMPEDLANGEVINPROPAGATOR_H_ */
