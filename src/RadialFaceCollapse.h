/*
 * RadialFaceCollapse.h
 *
 *  Created on: Feb 20, 2018
 *      Author: andy
 */

#ifndef SRC_RADIALFACECOLLAPSE_H_
#define SRC_RADIALFACECOLLAPSE_H_

#include "MeshOperations.h"

struct RadialFaceCollapse : MeshOperation {
public:

    /**
     * Creates a new vertex split mesh operation if
     * the operation is valid for the given vertex pointer.
     *
     * This looks at the six neighboring partial triangles to this
     * triangle, and checks the angle they make to this face.
     */
    static MeshOperation *create(MeshPtr mesh, TrianglePtr tri);



    virtual ~RadialFaceCollapse();



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

    /**
     * mark the edge, for debug purposes
     */
    virtual void mark() const;

private:

    RadialFaceCollapse(MeshPtr, CellPtr cell, PTrianglePtr a, PTrianglePtr b);


    float angleCutoff;
    Edge edge;
};

#endif /* SRC_RADIALFACECOLLAPSE_H_ */
