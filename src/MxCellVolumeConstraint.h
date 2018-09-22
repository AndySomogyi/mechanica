/*
 * MxCellVolumeConstraint.h
 *
 *  Created on: Sep 20, 2018
 *      Author: andy
 */

#ifndef SRC_MXCELLVOLUMECONSTRAINT_H_
#define SRC_MXCELLVOLUMECONSTRAINT_H_

#include "MxConstraints.h"
#include "MxCell.h"

struct MxCellVolumeConstraint : IConstraint
{
    MxCellVolumeConstraint(float targetVolume, float lambda);

    virtual HRESULT setTime(float time);

    virtual float energy(const MxObject **objs, int32_t len);

    virtual HRESULT project(MxObject **obj, int32_t len);

    float targetVolume;
    float lambda;

private:
    float energy(const MxObject *obj);
};

#endif /* SRC_MXCELLVOLUMECONSTRAINT_H_ */
