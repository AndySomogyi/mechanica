/*
 * MxEdge.h
 *
 *  Created on: Sep 27, 2017
 *      Author: andy
 */

#ifndef _INCLUDE_MXEDGE_H_
#define _INCLUDE_MXEDGE_H_

#include "MxCell.h"

struct MxEdge {
    MxVertex *a;
    MxVertex *b;

    typedef std::vector<MxFacet*> FacetVector;

    MxEdge(VertexPtr a, VertexPtr b);

    /**
     * find the edge that connects a pair of triangles. If these
     * triangles don't share an edge, both vertices are null.
     */
    MxEdge(const TrianglePtr a, const TrianglePtr b);

    EdgeFacets facets() const;

    std::vector<TrianglePtr> radialTriangles() const;


    const FacetVector& upperFacets() const;

    const FacetVector& lowerFacets() const;

    const FacetVector& radialFacets() const;

    bool operator == (const MxEdge& other);

    bool incidentTo(const MxTriangle& tri);

private:
    FacetVector upper, lower, radial;
};



#endif /* _INCLUDE_MXEDGE_H_ */
