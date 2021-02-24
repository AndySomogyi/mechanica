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

enum FluxKind {
    FLUX_FICK = 0,
    FLUX_SECRETE = 1,
    FLUX_UPTAKE = 2
};


// keep track of the ids of the particle types, to determine
// the reaction direction.
struct TypeIdPair {
    int16_t a;
    int16_t b;
};

struct MxFlux {
    int32_t       size; // temporary size until we get SIMD instructions.
    int8_t        kinds[MX_SIMD_SIZE];
    TypeIdPair    type_ids[MX_SIMD_SIZE];
    int32_t       indices_a[MX_SIMD_SIZE];
    int32_t       indices_b[MX_SIMD_SIZE];
    float         coef[MX_SIMD_SIZE];
    float         decay_coef[MX_SIMD_SIZE];
    float         target[MX_SIMD_SIZE];
};


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
    int32_t size;          // how many individual flux objects this has
    int32_t fluxes_size;   // how many fluxes (blocks) this has.
    static int32_t init;
    MxFlux fluxes[];       // allocated in single block, this
};

MxFluxes *MxFluxes_New(int32_t init_size);


MxFluxes *MxFluxes_AddFlux(FluxKind kind, MxFluxes *fluxes,
                           int16_t typeId_a, int16_t typeId_b,
                           int32_t index_a, int32_t index_b,
                           float k, float decay, float target);


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
PyObject* MxFluxes_Fick(PyObject *self, PyObject *args, PyObject *kwargs);

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
PyObject* MxFluxes_Secrete(PyObject *self, PyObject *args, PyObject *kwargs);

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
PyObject* MxFluxes_Uptake(PyObject *self, PyObject *args, PyObject *kwargs);


/**
 * integrate all of the fluxes for a space cell.
 */
HRESULT MxFluxes_Integrate(int cellId);

CAPI_DATA(PyTypeObject) MxFluxes_Type;

#endif /* SRC_MDCORE_SRC_FLUX_H_ */
