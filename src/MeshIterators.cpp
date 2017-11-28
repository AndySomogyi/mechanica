/*
 * MeshIterators.cpp
 *
 *  Created on: Aug 13, 2017
 *      Author: andy
 */

#include "MxCell.h"
#include <iostream>
#include <set>

static bool commonCell(const TrianglePtr a, const TrianglePtr b) {
    assert(a->cells[0] && a->cells[1] && b->cells[0] && b->cells[1]);
    return (a->cells[0] == b->cells[0] ||
            a->cells[0] == b->cells[1] ||
            a->cells[1] == b->cells[0] ||
            a->cells[1] == b->cells[1]);
}




EdgeTriangleIterator::value_type EdgeTriangleIterator::operator *() const {
    TrianglePtr tri = triangles[index];
    return tri;
}

EdgeTriangleIterator& EdgeTriangleIterator::operator ++() {
    index++;
    return *this;
}

EdgeTriangleIterator EdgeTriangleIterator::operator ++(int) {
    EdgeTriangleIterator ret = *this;
    ++(*this);
    return ret;
}

EdgeTriangleIterator& EdgeTriangleIterator::operator --() {
    index--;
    return *this;
}

EdgeTriangleIterator EdgeTriangleIterator::operator --(int) {
    EdgeTriangleIterator ret = *this;
    --(*this);
    return ret;
}

bool EdgeTriangleIterator::operator ==(
        const EdgeTriangleIterator& rhs) {
    return index == rhs.index && &triangles == &rhs.triangles;
}

bool EdgeTriangleIterator::operator !=(
        const EdgeTriangleIterator& rhs) {
    return index != rhs.index || &triangles != &rhs.triangles;
}

EdgeTriangles::iterator EdgeTriangles::begin() const {
    return EdgeTriangleIterator(triangles, 0);
}

EdgeTriangles::iterator EdgeTriangles::end() const {
    return EdgeTriangleIterator(triangles, triangles.size());
}

static Edge tri_to_edge(const TrianglePtr startTri, int index) {
    assert(index >= 0 && index <= 3);
    VertexPtr a = startTri->vertices[index];
    VertexPtr b = startTri->vertices[(index+1)%3];
    return {{a, b}};
}

EdgeTriangles::EdgeTriangles(const TrianglePtr startTri, int index) :
        EdgeTriangles{tri_to_edge(startTri, index)} {
}


EdgeTriangles::EdgeTriangles(const Edge& edge) {

    assert(edge[0] && edge[1]);

    for(TrianglePtr ta : edge[0]->triangles()) {
        if(incident(ta, edge[1])) {
            triangles.push_back(ta);
        }
    }

#ifndef NDEBUG
    static int ctr = 0;
#endif

    // TODO: TOTAL FUCKING HACK
    // we desperately need to come up with a cleaner way of representing
    // ordered triangles around an edge. The correct way to do this is with
    // radial edge pointers around each triangle. But, do that in the next
    // release.

    // need to sort the radial triangles, so each tri shares a cell with the next one.
    for(int i = 0; (i + 1) < triangles.size(); ++i) {
        if(commonCell(triangles[i], triangles[i+1])) continue;

        ctr++;

        // if we get here, that means that that triangles[i] and triangles[i+1]
        // do not share a common cell, so we need to find the common cell with
        // triangles[i], and stick it in the triangles[i+1] slot.
        for(uint j = i + 2; j < triangles.size(); ++j) {
            if(commonCell(triangles[i], triangles[j])) {
                std::swap(triangles[i+1], triangles[j]);
            }
        }

#ifndef NDEBUG
        if(!commonCell(triangles[i], triangles[i+1])) {
            for(int k = 0; k < triangles.size(); ++k) {

                std::cout << "tri[" << k << "], cells: {" <<
                triangles[k]->cells[0]->id << "," <<
                triangles[k]->cells[1]->id << "}" << std::endl;
            }
            std::cout << "foo" << std::endl;
        }
        assert(commonCell(triangles[i], triangles[i+1]));
#endif
    }
}


EdgeFacetIterator::EdgeFacetIterator(const class EdgeFacets& edgeStar) {
}

EdgeFacetIterator::reference EdgeFacetIterator::operator *() const {
}

EdgeTriangleIterator& EdgeFacetIterator::operator ++() {
}

EdgeTriangleIterator EdgeFacetIterator::operator ++(int int1) {
}

EdgeTriangleIterator& EdgeFacetIterator::operator --() {
}

EdgeTriangleIterator EdgeFacetIterator::operator --(int int1) {
}

bool EdgeFacetIterator::operator ==(const EdgeFacetIterator& rhs) {
}

bool EdgeFacetIterator::operator !=(const EdgeFacetIterator& rhs) {
}

EdgeFacets::const_iterator EdgeFacets::begin() const {
}

EdgeFacets::const_iterator EdgeFacets::end() const {
}

EdgeFacets::EdgeFacets(const TrianglePtr startTri,
        const std::array<VertexPtr, 2>& edge) {
}

size_t EdgeTriangles::size() const {
    return triangles.size();
}

#ifndef NDEBUG
bool EdgeTriangles::isValid() {
    std::set<CellPtr> cells;
    for(TrianglePtr tri : triangles) {
        for(int i = 0; i < 2; ++i) {
            if(tri->cells[i]) {
                cells.insert(tri->cells[i]);
            } else {
                std::cout << "triangle has null cell" << std::endl;
                return false;
            }
        }
    }
    bool result = cells.size() == triangles.size();
    if(!result) {
        std::cout << "cell count different than triangle count" << std::endl;
    }
    return true;
}
#endif
