/*
 * SkeletalEdgeToTriangleOperation.h
 *
 *  Created on: Jul 24, 2018
 *      Author: andy
 */

#ifndef SRC_SKELETALEDGETOTRIANGLEOPERATION_H_
#define SRC_SKELETALEDGETOTRIANGLEOPERATION_H_

#include <MeshOperations.h>

class SkeletalEdgeToTriangleOperation: public MeshOperation {

    public:

    static HRESULT create(MeshPtr, CSkeletalEdgePtr, SkeletalEdgeToTriangleOperation **result);


    private:
    SkeletalEdgeToTriangleOperation(MeshPtr, CSkeletalEdgePtr);
};

#endif /* SRC_SKELETALEDGETOTRIANGLEOPERATION_H_ */
