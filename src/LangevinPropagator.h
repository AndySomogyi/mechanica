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

    LangevinPropagator();
    
    /**
     * Attaches model to this propagator
     */
    HRESULT setModel(MxModel *model);

    HRESULT step(MxReal dt);

    /**
     * Inform the propagator that the model structure changed.
     */
    HRESULT structureChanged();


    HRESULT bindConstraint(IConstraint *constraint, CObject *obj);

    HRESULT bindForce(IForce *force, CObject *obj);

    HRESULT unbindConstraint(IConstraint* constraint);

    HRESULT unbindForce(IForce *force);



private:

    struct ConstraintItems {
        IConstraint *thing;
        CType *type;
        std::vector<CObject*> args;
    };

    struct ForceItems {
        IForce *thing;
        CType *type;
        std::vector<CObject*> args;
    };

    HRESULT applyForces();


    HRESULT eulerStep(MxReal dt);

    HRESULT rungeKuttaStep(MxReal dt);


    HRESULT getAccelerations(float time, uint32_t len, const Vector3 *pos, Vector3 *acc);

    //HRESULT getMasses(float time, uint32_t len, float *masses);

    HRESULT getPositions(float time, uint32_t len, Vector3 *pos);

    HRESULT setPositions(float time, uint32_t len, const Vector3 *pos);

    HRESULT applyConstraints();
    
    /**
     * The model structure changed, so we need to update all the
     * constraints
     */



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

    static HRESULT objectDeleteListener(CObject* pThis,
            const CObject* obj, uint32_t what);

    template<typename T>
    HRESULT updateItems(std::vector<T> &items);

    template<typename T>
    HRESULT updateItem(T &item);

    template<typename T, typename KeyType>
    T& getItem(std::vector<T> &items, KeyType *key);

    template<typename T, typename KeyType>
    HRESULT bindTypeItem(std::vector<T> &items, KeyType *key, CType* type);

};

HRESULT MxBind_PropagatorModel(LangevinPropagator *propagator, MxModel *model);



#endif /* SRC_MESHDAMPEDLANGEVINPROPAGATOR_H_ */
