/*
 * RadialEdgeCollapse.h
 *
 *  Created on: Nov 23, 2017
 *      Author: andy
 */

#ifndef SRC_RADIALEDGECOLLAPSE_H_
#define SRC_RADIALEDGECOLLAPSE_H_

#include "MeshOperations.h"


struct RadialEdgeCollapse : MeshOperation {

    RadialEdgeCollapse(MeshPtr, float shortCutoff, const Edge&);

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
    float shortCutoff;
    Edge edge;
};


#endif /* SRC_RADIALEDGECOLLAPSE_H_ */
