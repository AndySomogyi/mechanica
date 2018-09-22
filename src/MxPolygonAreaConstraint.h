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
    virtual float energy(const MxObject *obj);

    virtual HRESULT project(MxObject *obj);

};

#endif /* SRC_MXPOLYGONAREACONSTRAINT_H_ */
