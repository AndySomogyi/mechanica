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
    Magnum::Vector3 size;
};

struct MxCuboidHandle : PyObject
{
    int32_t id;
};

/**
 * check if a handle is a cublid
 */
CAPI_FUNC(int) MxCuboid_Check(PyObject *obj);

/**
 * check if a object is a cuboid type
 */
CAPI_FUNC(int) MxCuboidType_Check(PyObject *obj);

/**
 * vertex is special, it extends particle.
 */
CAPI_DATA(PyTypeObject) MxCuboid_Type;

HRESULT _MxCuboid_Init(PyObject *m);

void MxCuboid_UpdateAABB(MxCuboid *c);

#endif /* SRC_MDCORE_CUBOID_H_ */
