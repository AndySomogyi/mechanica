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

#define NEW_EDGE_TRIANGLE



EdgeTriangles::EdgeTriangles(const Edge& edge) {

    assert(edge[0] && edge[1]);

#ifdef NEW_EDGE_TRIANGLE

    TrianglePtr first = nullptr, prev = nullptr, tri = nullptr;

    for(TrianglePtr ta : edge[0]->triangles()) {
        if(incident(ta, edge[1])) {
            first = ta;
            triangles.push_back(first);
            break;
        }
    }

    tri = first->adjacentTriangleForEdge(edge[0], edge[1]);
    prev = first;

    assert(tri);
    assert(tri != first);

    do {
        triangles.push_back(tri);
        TrianglePtr next = tri->nextTriangleInRing(prev);
#ifndef NDEBUG
        if(!next) {
            std::cout << "error, EdgeTriangles, tri->nextTriangleInRing(prev) -> null, " << std::endl
            << "tri: " << tri << std::endl
            << "prev: " << prev << std::endl;
            tri->nextTriangleInRing(prev);
            assert(0);
        }
#endif
        assert(next);
        prev = tri;
        tri = next;
    } while(tri != first);
    
#ifndef NDEBUG
    for(int i = 0; i < triangles.size(); ++i) {
        TrianglePtr tri = triangles[i];
        assert(incident(tri, edge));
        if(i+1 < triangles.size()) {
            assert(commonCell(triangles[i], triangles[i+1]));
        }
    }
#endif




#else

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

#ifndef NDEBUG
        ctr++;
#endif

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

#endif //  NEW_EDGE_ITERATOR
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

std::vector<TrianglePtr> triangleFan(CVertexPtr vert, CCellPtr cell)
{
    std::vector<TrianglePtr> fan;

    // get the first triangle
    TrianglePtr first = vert->triangleForCell(cell);

    if(!first) {
        return fan;
    }

    // the loop triangle
    TrianglePtr tri = first;
    // keep track of the previous triangle
    TrianglePtr prev = nullptr;
    do {
        assert(incident(tri, vert));
        fan.push_back(tri);
        TrianglePtr next = tri->nextTriangleInFan(vert, cell, prev);
        prev = tri;
        tri = next;
    } while(tri && tri != first);

    return fan;
}

#endif
