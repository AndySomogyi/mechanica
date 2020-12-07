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


struct CAPI_EXPORT MxUniverse  {

    static Magnum::Vector3 origin();

    static  Magnum::Vector3 dim();


    bool isRunning;

    unsigned performance_info_display_interval;

    uint32_t performance_info_flags;
};

/**
 *
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
 * int engine_init ( struct engine *e , const double *origin , const double *dim , double *L ,
        double cutoff , unsigned int period , int max_type , unsigned int flags );
 */

struct CAPI_EXPORT MxUniverseConfig {
    Magnum::Vector3 origin;
    Magnum::Vector3 dim;
    Magnum::Vector3i spaceGridSize;
    Magnum::Vector3ui boundaryConditions;
    double cutoff;
    uint32_t flags;
    uint32_t maxTypes;
    double dt;
    double temp;
    int nParticles;
    int threads;
    EngineIntegrator integrator;
    uint32_t periodic;
    double max_distance;
    MxUniverseConfig();
};

CAPI_FUNC(HRESULT) MxUniverse_Init(const MxUniverseConfig &conf);

CAPI_FUNC(HRESULT) MxUniverse_Bind(PyObject *args, PyObject *kwargs, PyObject **result);

CAPI_FUNC(HRESULT) MxUniverse_BindThing3(PyObject *thing, PyObject *a, PyObject *b, PyObject *c);

CAPI_FUNC(HRESULT) MxUniverse_BindThing2(PyObject *thing, PyObject *a, PyObject *b);

CAPI_FUNC(HRESULT) MxUniverse_BindThing1(PyObject *thing, PyObject *a);


/**
 * generate a surface mesh and bind it with a potential.
 *
 * args:
 *     potential
 *     number of subdivisions
 *     tuple of starting / stopping theta (polar angle)
 *     center of sphere
 *     radius of sphere
 */
CAPI_FUNC(PyObject*) MxUniverse_BindSphere(PyObject *thing, PyObject *a);

PyObject *MxPyUniverse_BindPairwise(PyObject *_args, PyObject *_kwargs);

/**
 * runs the universe a pre-determined period of time, until.
 * can use micro time steps of 'dt' which override the
 * saved universe dt.
 *
 * if until is 0, it is ignored and the universe.dt is used.
 * if dt is 0, it is ignored, and the universe.dt is used as
 * a single time step.
 */
CAPI_FUNC(HRESULT) MxUniverse_Step(double until, double dt);


/**
 * starts the universe time evolution. The simulator
 * actually advances the universe, this method just
 * tells the simulator to perform the time evolution.
 */
enum MxUniverse_Flags {
    MX_RUNNING = 1 << 0,

    MX_SHOW_PERF_STATS = 1 << 1,

    // in ipython message loop, monitor console
    MX_IPYTHON_MSGLOOP = 1 << 2,

    // standard polling message loop
    MX_POLLING_MSGLOOP = 1 << 3,
};

/**
 * get a flag value
 */
CAPI_FUNC(int) MxUniverse_Flag(MxUniverse_Flags flag);

/**
 * sets / clears a flag value
 */
CAPI_FUNC(HRESULT) MxUniverse_SetFlag(MxUniverse_Flags flag, int value);



/**
 * The single global instance of the universe
 */
CAPI_DATA(MxUniverse) Universe;


/**
 * Init and add to python module
 */
HRESULT _MxUniverse_init(PyObject *m);



#endif /* SRC_MXUNIVERSE_H_ */
