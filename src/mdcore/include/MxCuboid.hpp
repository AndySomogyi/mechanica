/*
 * MxCuboid.h
 *
 *  Created on: Jan 17, 2021
 *      Author: andy
 */
#pragma once
#ifndef SRC_MDCORE_CUBOID_H_
#define SRC_MDCORE_CUBOID_H_

#include <platform.h>
#include <MxBody.hpp>
#include <Magnum/Math/Quaternion.h>

struct MxCuboid : MxBody
{
    MxCuboid();
    
    // extents / size of the cuboid
    Magnum::Vector3 extents;
    
    Magnum::Quaternion rotation;
    

};

struct MxCuboidHandle : PyObject
{
    int32_t id;
};

/**
 * vertex is special, it extends particle.
 */
CAPI_DATA(MxParticleType*) MxCuboid_TypePtr;

HRESULT _MxCuboid_Init(PyObject *m);

#endif /* SRC_MDCORE_CUBOID_H_ */
