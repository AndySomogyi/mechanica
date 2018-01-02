/*
 * VertexSplit.h
 *
 *  Created on: Dec 15, 2017
 *      Author: andy
 */

#ifndef SRC_VERTEXSPLIT_H_
#define SRC_VERTEXSPLIT_H_

#include "MeshOperations.h"

class VertexSplit : public MeshOperation {
public:

    /**
     * Creates a new vertex split mesh operation if
     * the operation is valid for the given vertex pointer.
     */
    static MeshOperation *create(MeshPtr mesh, VertexPtr);

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
    virtual bool depends(CTrianglePtr) const;

    /**
     * does this operation depend on this vertex?
     */
    virtual bool depends(CVertexPtr) const;

    virtual bool equals(const Edge& e) const;
    
    virtual bool equals(CVertexPtr) const;

    virtual void mark() const;

private:

    VertexSplit(MeshPtr, VertexPtr);

    VertexPtr vertex;
    
    const int id = 0;
};

#endif /* SRC_VERTEXSPLIT_H_ */
