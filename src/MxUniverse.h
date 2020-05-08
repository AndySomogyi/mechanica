/*
 * MxUniverse.h
 *
 *  Created on: Sep 7, 2019
 *      Author: andy
 */

#ifndef SRC_MXUNIVERSE_H_
#define SRC_MXUNIVERSE_H_

#include "mechanica_private.h"
#include "mdcore_single.h"


struct MxUniverse  {


    /**
     * MDCore MD engine
     */
    CListWrap potentials;

    static Magnum::Vector3 origin();

    static  Magnum::Vector3 dim();
};

/**
 * /**
 * @brief Initialize an #engine with the given data.
 *
 * The number of spatial cells in each cartesion dimension is floor( dim[i] / L[i] ), or
 * the physical size of the space in that dimension divided by the minimum size size of
 * each cell.
 *
 * @param e The #engine to initialize.
 * @param origin An array of three doubles containing the cartesian origin
 *      of the space.
 * @param dim An array of three doubles containing the size of the space.
 *
 * @param L The minimum spatial cell edge length in each dimension.
 *
 * @param cutoff The maximum interaction cutoff to use.
 * @param period A bitmask describing the periodicity of the domain
 *      (see #space_periodic_full).
 * @param max_type The maximum number of particle types that will be used
 *      by this engine.
 * @param flags Bit-mask containing the flags for this engine.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 *
*int engine_init ( struct engine *e , const double *origin , const double *dim , double *L ,
        double cutoff , unsigned int period , int max_type , unsigned int flags );
 */

struct MxUniverseConfig {
    Magnum::Vector3 origin;
    Magnum::Vector3 dim;
    Magnum::Vector3i spaceGridSize;
    Magnum::Vector3ui boundaryConditions;
    double cutoff;
    uint32_t flags;
    uint32_t maxTypes;

    MxUniverseConfig();
};




/**
 * The single global instance of the universe
 */
CAPI_DATA(MxUniverse) Universe;

/**
 * Init and add to python module
 */
HRESULT MxUniverse_init(PyObject *m);



#endif /* SRC_MXUNIVERSE_H_ */
