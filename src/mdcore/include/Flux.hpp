/*
 * Flux.h
 *
 *  Created on: Dec 21, 2020
 *      Author: andy
 */

#ifndef SRC_MDCORE_SRC_FLUX_H_
#define SRC_MDCORE_SRC_FLUX_H_

#include "platform.h"
#include "mdcore_single_config.h"


/**
 * flux is defined btween a pair of types, and acts on the
 * state vector between a pair of instances.
 *
 * The indices of the species in each state vector
 * are most likely different, so we keep track of the
 * indices in each type, and the transport constatants.
 *
 * A flux between a pair of types, and pair of respective
 * species need:
 *
 * (1) type A, (2) type B, (3) species id in A, (4) species id in B,
 * (5) transport constant.
 *
 * aloocate Flux as a single block, member pointers point to
 * offsets in these blocks.
 */
struct MxFluxes : PyVarObject
{
    int32_t *indices_a;
    int32_t *indices_b;
    float *coef;
    
    static int32_t init;
};


CAPI_DATA(PyTypeObject) MxFluxes_Type;

#endif /* SRC_MDCORE_SRC_FLUX_H_ */
