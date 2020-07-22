/*
 * MxReactiveBond.h
 *
 *  Created on: Jul 16, 2020
 *      Author: andy
 */

#ifndef SRC_MDCORE_INCLUDE_MXBONDGENERATOR_H_
#define SRC_MDCORE_INCLUDE_MXBONDGENERATOR_H_

#include "MxPotential.h"

typedef HRESULT (*reactivepotential_invoke)(struct MxReactivePotential *,
        struct MxParticle *a, struct MxParticle *b);

/** The #potential structure. */
typedef struct MxReactivePotential : MxPotential {
    double activation_energy;

    double activation_distance;


    /**
     * Bond potential, this is what gets copied to the bond,
     * not used is non-bonded potential.
     */
    MxPotential *bond_potential;

    /**
     * Function that gets invoked when the potential is triggered
     * (energy exceeds threshold).
     */
    reactivepotential_invoke invoke;

} MxReactiveBond;

CAPI_FUNC(HRESULT) MxBondGenerator();


/**
 * The type of each individual particle.
 */
CAPI_DATA(PyTypeObject) MxReactivePotential_Type;


// internal init TODO: move this to internal header
HRESULT _MxReactivePotential_init(PyObject *m);





#endif /* SRC_MDCORE_INCLUDE_MXBONDGENERATOR_H_ */
