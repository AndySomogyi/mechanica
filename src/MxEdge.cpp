/*
 * MxEdge.cpp
 *
 *  Created on: Sep 27, 2017
 *      Author: andy
 */

#include "MxCell.h"
#include "MxEdge.h"
#include <algorithm>

static bool commonCell(const TrianglePtr a, const TrianglePtr b) {
    assert(a->cells[0] && a->cells[1] && b->cells[0] && b->cells[1]);
    return (a->cells[0] == b->cells[0] ||
            a->cells[0] == b->cells[1] ||
            a->cells[1] == b->cells[0] ||
            a->cells[1] == b->cells[1]);
}

MxEdge::MxEdge(VertexPtr a, VertexPtr b) :
a{a}, b{b} {
    len = (a->position - b->position).length();

    int triCnt = 0;
    for(TrianglePtr t : a->triangles()) {
        if(contains(b->triangles(), t)) {
            triCnt += 1;
        }
    }
    assert(triCnt > 0);

    for(FacetPtr fa : a->facets()) {
        if(incident(fa, b)) {
            radial.push_back(fa);
        } else {
            upper.push_back(fa);
        }
    }

    for(FacetPtr fb : b->facets()) {
        if(!incident(fb, a)) {
            lower.push_back(fb);
        } else {
            assert(contains(radial, fb));
        }
    }

    for(TrianglePtr ta : a->triangles()) {
        if(incident(ta, b)) {
            radialTri.push_back(ta);
        }
    }

    // TODO: TOTAL FUCKING HACK
    // we desperately need to come up with a cleaner way of representing
    // ordered triangles around an edge. The correct way to do this is with
    // radial edge pointers around each triangle. But, do that in the next
    // release.

    // need to sort the radial triangles, so each tri shares a cell with the next one.
    for(int i = 0; (i + 1) < radialTri.size(); ++i) {
        if(commonCell(radialTri[i], radialTri[i+1])) continue;

        for(uint j = i + 2; j < radialTri.size(); ++j) {
            if(commonCell(radialTri[i], radialTri[j])) {
                std::swap(radialTri[i+1], radialTri[j]);
            }
        }

        assert(commonCell(radialTri[i], radialTri[i+1]));
    }
}

//MxEdge::MxEdge(const TrianglePtr a, const TrianglePtr b)
//{
//    //len = (a->positionsCh))
//}

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

std::set<VertexPtr> MxEdge::link() const {
    std::set<VertexPtr> lnk;
    for(TrianglePtr tri : radialTri) {
        for(VertexPtr v : tri->vertices) {
            lnk.insert(v);
        }
    }
    return lnk;
}
