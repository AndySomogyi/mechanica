/*
 * MxEdge.h
 *
 *  Created on: Sep 27, 2017
 *      Author: andy
 */

#ifndef _INCLUDE_MXEDGE_H_
#define _INCLUDE_MXEDGE_H_

#include "MxCell.h"
#include <array>


/**
 * An edge connects two vertices, and an edge must be incident to at
 * least one triangle. An edge can be incident to 1 ... n triangles.
 *
 * If a facet contains only the top vertex a, but not the bottom one,
 * then that facet in in the set of top facets. Similarly, if a facet only
 * contains the bottom vertex b, but not the top, then it is a bottom
 * facet. If a facet contains both vertices, then it is a radial facet.
 */
struct MxEdge {
    MxVertex *a;
    MxVertex *b;

    typedef std::vector<FacetPtr> FacetVector;
    typedef std::vector<TrianglePtr> TriangleVector;

    MxEdge(VertexPtr a, VertexPtr b);
    
    MxEdge(const Edge& e) : MxEdge{e[0], e[1]} {};

    /**
     * find the edge that connects a pair of triangles. If these
     * triangles don't share an edge, both vertices are null.
     */
    MxEdge(const TrianglePtr a, const TrianglePtr b);

    const TriangleVector& radialTriangles() const;

    const FacetVector& upperFacets() const;

    const FacetVector& lowerFacets() const;

    const FacetVector& radialFacets() const;

    bool operator == (const MxEdge& other) const;

    bool operator < (const MxEdge& other) const;

    bool operator > (const MxEdge& other) const;

    bool operator == (const std::array<VertexPtr, 2> &verts) const;
    
    float length() const {
        return (a->position - b->position).length();
    }


    bool incidentTo(const MxTriangle& tri) const;

private:
    FacetVector upper, lower, radial;
    TriangleVector radialTri    ;
    float len;
};



#endif /* _INCLUDE_MXEDGE_H_ */
