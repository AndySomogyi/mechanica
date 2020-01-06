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

    virtual float energy(const CObject **objs, int32_t len);

    virtual HRESULT project(CObject **obj, int32_t len);

    float targetArea;
    float lambda;

    float energy(const CObject* obj);
};

#endif /* SRC_MXPOLYGONAREACONSTRAINT_H_ */
