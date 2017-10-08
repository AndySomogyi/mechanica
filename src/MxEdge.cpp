/*
 * MxEdge.cpp
 *
 *  Created on: Sep 27, 2017
 *      Author: andy
 */

#include "MxCell.h"
#include "MxEdge.h"

MxEdge::MxEdge(VertexPtr a, VertexPtr b) :
a{a}, b{b} {
    len = (a->position - b->position).length();

    for(FacetPtr fa : a->facets) {
        if(incident(fa, b)) {
            radial.push_back(fa);
        } else {
            upper.push_back(fa);
        }
    }

    for(FacetPtr fb : b->facets) {
        if(!incident(fb, a)) {
            lower.push_back(fb);
        } else {
            assert(contains(radial, fb));
        }
    }

    for(TrianglePtr ta : a->triangles) {
        if(incident(ta, b)) {
            radialTri.push_back(ta);
        }
    }
}

MxEdge::MxEdge(const TrianglePtr a, const TrianglePtr b)
{
    //len = (a->positionsCh))
}

//EdgeFacets MxEdge::facets() const {
//}
const MxEdge::TriangleVector& MxEdge::radialTriangles() const {
    return radialTri;
}

const MxEdge::FacetVector& MxEdge::upperFacets() const {
    return upper;
}

const MxEdge::FacetVector& MxEdge::lowerFacets() const {
    return lower;
}

const MxEdge::FacetVector& MxEdge::radialFacets() const {
    return radial;
}

bool MxEdge::operator ==(const MxEdge& other) const {
    return (a == other.a && b == other.b) || (a == other.b && b == other.a);
}

bool MxEdge::operator ==(const std::array<VertexPtr, 2>& verts) const {
    return (a == verts[0] && b == verts[1]) || (b == verts[0] && a == verts[1]);
}

bool MxEdge::operator <(const MxEdge& other) const {
    return len < other.len;
}

bool MxEdge::incidentTo(const MxTriangle& tri) const {
}

bool MxEdge::operator >(const MxEdge& other) const {
    return len > other.len;
}
