/*
 * MxPolygonAreaConstraint.h
 *
 *  Created on: Sep 20, 2018
 *      Author: andy
 */

#ifndef SRC_MXPOLYGONAREACONSTRAINT_H_
#define SRC_MXPOLYGONAREACONSTRAINT_H_

#include "MxConstraints.h"

struct MxPolygonAreaConstraint : IConstraint
{
    MxPolygonAreaConstraint(float targetArea, float lambda);

    virtual HRESULT setTime(float time);

    virtual float energy(const MxObject **objs, int32_t len);

    virtual HRESULT project(MxObject **obj, int32_t len);

    float targetArea;
    float lambda;
};

#endif /* SRC_MXPOLYGONAREACONSTRAINT_H_ */
