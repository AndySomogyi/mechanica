/*
 * MxForces.h
 *
 *  Created on: Sep 19, 2018
 *      Author: andy
 */

#ifndef SRC_MXFORCES_H_
#define SRC_MXFORCES_H_

#include <carbon.h>
#include "mechanica_private.h"


/**
 * Interface that force objects implement.
 *
 * Forces can be time-dependent, and contain state variables. Forces can be stepped
 * in time just like physical objects.
 */
struct IForce {

    /**
     * Called when the main time step changes.
     */
    virtual HRESULT setTime(float time) = 0;

    /**
     * Apply forces to a set of objects.
     */
    virtual HRESULT applyForce(float time, CObject **objs, uint32_t len) const = 0;
};


struct MxForces
{
};

#endif /* SRC_MXFORCES_H_ */
