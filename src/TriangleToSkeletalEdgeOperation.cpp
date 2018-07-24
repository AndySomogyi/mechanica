/*
 * TriangleToSkeletalEdgeOperation.cpp
 *
 *  Created on: Jul 24, 2018
 *      Author: andy
 */

#include <TriangleToSkeletalEdgeOperation.h>

HRESULT TriangleToSkeletalEdgeOperation::create(MeshPtr, CTrianglePtr,
        TriangleToSkeletalEdgeOperation** result) {
    return E_FAIL;
}

TriangleToSkeletalEdgeOperation::TriangleToSkeletalEdgeOperation(MeshPtr mesh,
    CTrianglePtr) :
    MeshOperation{mesh} {
}
