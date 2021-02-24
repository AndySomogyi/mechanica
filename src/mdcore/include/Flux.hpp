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
 *
 * Allocated size is:
 * sizeof(MxFluxes) + 2 * alloc_size * sizeof(int32) + alloc_size * sizeof(float)
 */
struct MxFluxes : PyObject
{
    int32_t size;          // size of how many fluxes this object has
    int32_t alloc_size;    // size of how much space we have for fluxes.
    static int32_t init;
    
    int32_t *indices_a;
    int32_t *indices_b;
    float *coef;
    float *decay_coef;
};

MxFluxes *MxFluxes_New(int32_t init_size);


MxFluxes *MxFluxes_AddFlux(MxFluxes *fluxes, int32_t index_a, int32_t index_b, float k, float decay);


/**
 * The global mechanica.flux function.
 *
 * python interface to add fluxes
 *
 * args a:ParticleType, b:ParticleType, s:String, k:Float
 *
 * looks for a fluxes between types a and b, adds a flux for the
 * species named 's' with coef k.
 */
CAPI_FUNC(PyObject*) MxFluxes_FluxPy(PyObject *self, PyObject *args, PyObject *kwargs);


/**
 * integrate all of the fluxes for a space cell.
 */
HRESULT MxFluxes_Integrate(int cellId);

CAPI_DATA(PyTypeObject) MxFluxes_Type;

#endif /* SRC_MDCORE_SRC_FLUX_H_ */
