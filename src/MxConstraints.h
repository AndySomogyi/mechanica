/*
 * MxConstraints.h
 *
 *  Created on: Sep 19, 2018
 *      Author: andy
 */

#ifndef SRC_MXCONSTRAINTS_H_
#define SRC_MXCONSTRAINTS_H_

#include "mechanica_private.h"
#include <vector>

struct IConstraint {

    virtual HRESULT setTime(float time) = 0;

    virtual float energy(const MxObject **objs, int32_t len) = 0;

    virtual HRESULT project(MxObject **obj, int32_t len) = 0;
};


struct MxConstrainableType : MxType {


    std::vector<IConstraint*> constraints;

};

struct MxConstrainable {


};


class MxConstraints
{
};

#endif /* SRC_MXCONSTRAINTS_H_ */
