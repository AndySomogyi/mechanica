/*
 * RadialEdgeFlip.h
 *
 *  Created on: Nov 30, 2017
 *      Author: andy
 */

#ifndef SRC_RADIALEDGEFLIP_H_
#define SRC_RADIALEDGEFLIP_H_

#include "MeshOperations.h"

struct RadialEdgeFlip : MeshOperation {

    RadialEdgeFlip(MeshPtr msh, VertexPtr vert);

    static bool applicable(const Edge& e);

    /**
     * Apply this operation
     */
    virtual HRESULT apply();

    /**
     * lower, more negative energy operations are queued at a higher priority.
     */
    virtual float energy() const;

    /**
     * does this operation depend on this triangle?
     */
    virtual bool depends(const TrianglePtr) const;

    /**
     * does this operation depend on this vertex?
     */
    virtual bool depends(const VertexPtr) const;

    virtual bool equals(const Edge& e) const;

    virtual void mark() const;

private:
    VertexPtr vertex;

};

#endif /* SRC_RADIALEDGEFLIP_H_ */
