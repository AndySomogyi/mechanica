/*
 * SkeletalEdgeToTriangleOperation.cpp
 *
 *  Created on: Jul 24, 2018
 *      Author: andy
 */

#include <SkeletalEdgeToTriangleOperation.h>

HRESULT create(MeshPtr, CSkeletalEdgePtr, SkeletalEdgeToTriangleOperation **result) {
    return E_FAIL;
}

SkeletalEdgeToTriangleOperation::SkeletalEdgeToTriangleOperation(MeshPtr mesh,
    CSkeletalEdgePtr) :
    MeshOperation{mesh} {
}
