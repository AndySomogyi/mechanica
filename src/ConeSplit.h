/*
 * ConeSplit.h
 *
 *  Created on: Nov 30, 2017
 *      Author: andy
 */

#ifndef SRC_CONESPLIT_H_
#define SRC_CONESPLIT_H_

#include "MeshOperations.h"

struct ConeSplit : MeshOperation {

    ConeSplit(MeshPtr mesh, VertexPtr vert);

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
    VertexPtr centerVertex;
};

#endif /* SRC_CONESPLIT_H_ */
