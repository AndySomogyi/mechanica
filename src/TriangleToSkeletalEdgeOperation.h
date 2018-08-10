/*
 * TriangleToSkeletalEdgeOperation.h
 *
 *  Created on: Jul 24, 2018
 *      Author: andy
 */

#ifndef SRC_TRIANGLETOSKELETALEDGEOPERATION_H_
#define SRC_TRIANGLETOSKELETALEDGEOPERATION_H_

#include <MeshOperations.h>

class TriangleToSkeletalEdgeOperation: public MeshOperation {
public:
    HRESULT create (MeshPtr, CPolygonPtr, TriangleToSkeletalEdgeOperation **result);

private:
    TriangleToSkeletalEdgeOperation(MeshPtr, CPolygonPtr);
};

#endif /* SRC_TRIANGLETOSKELETALEDGEOPERATION_H_ */
