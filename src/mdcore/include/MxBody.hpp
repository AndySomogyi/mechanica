/*
 * MxBody.h
 *
 *  Created on: Jan 17, 2021
 *      Author: andy
 */
#pragma once
#ifndef SRC_MDCORE_BODY_H_
#define SRC_MDCORE_BODY_H_

#include <platform.h>
#include <MxParticle.h>

struct MxBody : PyObject
{
    /** Particle position */
    union {
        FPTYPE x[4] __attribute__ ((aligned (16)));
        Magnum::Vector3 position __attribute__ ((aligned (16)));

        struct {
            float __dummy[3];
            uint32_t creation_time;
        };
    };

    /** linear velocity */
    union {
        FPTYPE v[4] __attribute__ ((aligned (16)));
        Magnum::Vector3 velocity __attribute__ ((aligned (16)));
    };
    
    /**
     * linear force
     *
     * ONLY the coherent part of the force should go here. We use multi-step
     * integrators, that need to separate the random and coherent forces.
     */
    union {
        FPTYPE f[4] __attribute__ ((aligned (16)));
        Magnum::Vector3 force __attribute__ ((aligned (16)));
    };
    
    union {
        FPTYPE pad_orientation[4] __attribute__ ((aligned (16)));
        Magnum::Quaternion orientation __attribute__ ((aligned (16)));
    };
    
    /** angular velocity */
    union {
        FPTYPE _spin[4] __attribute__ ((aligned (16)));
        Magnum::Vector3 spin __attribute__ ((aligned (16)));
    };

    union {
        FPTYPE _torque[4] __attribute__ ((aligned (16)));
        Magnum::Vector3 torque __attribute__ ((aligned (16)));
    };
    
    /**
     * inverse rotation transform. 
     */
    Magnum::Quaternion inv_orientation;
    
    /**
     * update the aabb on motion. 
     */
    Magnum::Vector3 aabb_size;
        

    /** random force force */
    union {
        Magnum::Vector3 persistent_force __attribute__ ((aligned (16)));
    };

    // inverse mass
    double imass;

    double mass;
    
    // index of the object in some array, negative for invalid.
    int32_t id;

    /** Particle flags */
    uint32_t flags;

    /**
     * pointer to the python 'wrapper'. Need this because the particle data
     * gets moved around between cells, and python can't hold onto that directly,
     * so keep a pointer to the python object, and update that pointer
     * when this object gets moved.
     *
     * initialzied to null, and only set when .
     */
    PyObject *_handle;

    /**
     * public way of getting the pyparticle. Creates and caches one if
     * it's not there. Returns a inc-reffed handle, caller is responsible
     * for freeing it.
     */
    PyObject *handle();


    // style pointer, set at object construction time.
    // may be re-set by users later.
    // the base particle type has a default style.
    NOMStyle *style;

    /**
     * pointer to state vector (optional)
     */
    struct CStateVector *state_vector;
    
    MxBody();
};

struct MxBodyHandle : PyObject
{
    int32_t id;
};

/**
 * vertex is special, it extends particle.
 */
CAPI_DATA(PyTypeObject) MxBody_Type;

HRESULT _MxBody_Init(PyObject *m);

#endif /* SRC_MDCORE_BODY_H_ */
