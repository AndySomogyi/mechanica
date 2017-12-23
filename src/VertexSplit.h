/*
 * VertexSplit.h
 *
 *  Created on: Dec 15, 2017
 *      Author: andy
 */

#ifndef SRC_VERTEXSPLIT_H_
#define SRC_VERTEXSPLIT_H_

#include "MeshOperations.h"

class VertexSplit : MeshOperation {
    VertexSplit(MeshPtr, CVertexPtr);

    /**
     * Creates a new vertex split mesh operation if
     * the operation is valid for the given vertex pointer.
     */
    static MeshOperation *create(CVertexPtr);

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

    Edge edge;
};

#endif /* SRC_VERTEXSPLIT_H_ */
