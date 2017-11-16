/*
 * MeshIterators.cpp
 *
 *  Created on: Aug 13, 2017
 *      Author: andy
 */

#include <MxCell.h>

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

EdgeTriangles::EdgeTriangles(const TrianglePtr startTri, int index) {
    assert(index >= 0 && index <= 3);
    VertexPtr a = startTri->vertices[index];
    VertexPtr b = startTri->vertices[(index+1)%3];

    for(TrianglePtr ta : a->triangles()) {
        if(incident(ta, b)) {
            triangles.push_back(ta);
        }
    }

    // TODO: TOTAL FUCKING HACK
    // we desperately need to come up with a cleaner way of representing
    // ordered triangles around an edge. The correct way to do this is with
    // radial edge pointers around each triangle. But, do that in the next
    // release.

    // need to sort the radial triangles, so each tri shares a cell with the next one.
    for(int i = 0; (i + 1) < triangles.size(); ++i) {
        if(commonCell(triangles[i], triangles[i+1])) continue;

        for(uint j = i + 2; j < triangles.size(); ++j) {
            if(commonCell(triangles[i], triangles[j])) {
                std::swap(triangles[i+1], triangles[j]);
            }
        }

        assert(commonCell(triangles[i], triangles[i+1]));
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
