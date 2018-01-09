/*
 * MxTriangle.cpp
 *
 *  Created on: Oct 3, 2017
 *      Author: andy
 */

#include "MxTriangle.h"
#include "MxCell.h"
#include "MxDebug.h"
#include <iostream>

static std::string to_string(CCellPtr cell) {
    return cell ? std::to_string(cell->id) : "null";
}

static std::string to_string(const MxPartialTriangle *pt) {
    return "triId:" + (pt ? std::to_string(pt->triangle->id) : "null");
}

std::ostream& operator<<(std::ostream& os, CTrianglePtr tri)
{
    os << "Triangle {" << std::endl
       << "id:" << tri->id << "," << std::endl
       << "cells:{" << to_string(tri->cells[0]) << "," << to_string(tri->cells[1]) << "}," << std::endl
       << "vertices:{" << std::endl
       << "\t" << tri->vertices[0] << ", " << std::endl
       << "\t" << tri->vertices[1] << ", " << std::endl
       << "\t" << tri->vertices[2] << "}" << std::endl
       << "neighbors0:{" << to_string(tri->partialTriangles[0].neighbors[0]) << ","
                         << to_string(tri->partialTriangles[0].neighbors[1]) << ","
                         << to_string(tri->partialTriangles[0].neighbors[2]) << "}," << std::endl
       << "neighbors1:{" << to_string(tri->partialTriangles[1].neighbors[0]) << ","
                         << to_string(tri->partialTriangles[1].neighbors[1]) << ","
                         << to_string(tri->partialTriangles[1].neighbors[2]) << "}," << std::endl
       << "}" << std::endl;
    return os;
}



int MxTriangle::matchVertexIndices(const std::array<VertexPtr, 3> &indices) {
    typedef std::array<VertexPtr, 3> vertind;

    if (vertices == indices ||
        vertices == vertind{{indices[1], indices[2], indices[0]}} ||
        vertices == vertind{{indices[2], indices[0], indices[1]}}) {
        return 1;
    }

    if (vertices == vertind{{indices[2], indices[1], indices[0]}} ||
        vertices == vertind{{indices[1], indices[0], indices[2]}} ||
        vertices == vertind{{indices[0], indices[2], indices[1]}}) {
        return -1;
    }
    return 0;
}


MxTriangle::MxTriangle(uint _id, MxTriangleType* type,
        const std::array<VertexPtr, 3>& verts,
        const std::array<CellPtr, 2>& cells,
        const std::array<MxPartialTriangleType*, 2>& partTriTypes) :
            id{_id}, MxObject{type}, vertices{verts}, cells{cells},
            partialTriangles{{{partTriTypes[0], this}, {partTriTypes[1], this}}} {

    // connect this triangle to the vertex tri lists
    for(VertexPtr vert : verts) {
        auto res = vert->appendTriangle(this);
        assert(res==S_OK);
    }

    positionsChanged();
}

int MxTriangle::adjacentEdgeIndex(CVertexPtr a, CVertexPtr b) const {
    for(int i = 0; i < 3; ++i) {
        if((vertices[i] == a && vertices[(i+1)%3] == b) ||
           (vertices[i] == b && vertices[(i+1)%3] == a)) {
            return i;
        }
    }
    return -1;
}

HRESULT MxTriangle::positionsChanged() {

    const Vector3& v1 = vertices[0]->position;
    const Vector3& v2 = vertices[1]->position;
    const Vector3& v3 = vertices[2]->position;

    // the aspect ratio
    float a = (v1 - v2).length();
    float b = (v2 - v3).length();
    float c = (v3 - v1).length();
    float s = (a + b + c) / 2.0;
    aspectRatio = (a * b * c) / (8.0 * (s - a) * (s - b) * (s - c));

    // A surface normal for a triangle can be calculated by taking the vector cross product
    // of two edges of that triangle. The order of the vertices used in the calculation will
    // affect the direction of the normal (in or out of the face w.r.t. winding).

    // So for a triangle p1, p2, p3, if the vector U = p2 - p1 and the
    // vector V = p3 - p1 then the normal N = U x V and can be calculated by:

    // Nx = UyVz - UzVy
    // Ny = UzVx - UxVz
    // Nz = UxVy - UyVx
    // non-normalized normal vector
    // multiply by neg 1, CCW winding.
    Vector3 abnormal = Math::normal(v1, v2, v3);
    float len = abnormal.length();
    area = 0.5 * len;
    normal = abnormal / len;

    // average position of 3 position vectors
    centroid = (v1 + v2 + v3) / 3;

    // TODO: change vertex mass only in response to some sort of mass change
    // event -- we're mass conserving.
    for(int i = 0; i < 3; ++i) {
        vertices[i]->area += area / 3.;
        vertices[i]->mass += getMass() / 3.;
    }

    return S_OK;
}


bool MxTriangle::isConnected() const {

    for(int i = 0; i < 2; ++i) {

        if(cells[i]->isRoot()) continue;

        // TODO: HACK fix the constness.

        const PTrianglePtr t = const_cast<PTrianglePtr>(&partialTriangles[i]);

        bool padj0 = adjacent(t, t->neighbors[0]);
        bool padj1 = adjacent(t, t->neighbors[1]);
        bool padj2 = adjacent(t, t->neighbors[2]);
        bool padj = padj0 && padj1 && padj2;

        // check pointers
        if (!padj) {
            std::cout << "error, partial triangles neighbors not adjacent to this triangle" << std::endl;
            return false;
        }

        assert(this == t->triangle);

        // check vertices
        bool tadj0 = adjacent_triangle_vertices(const_cast<TrianglePtr>(this), t->neighbors[0]->triangle);
        bool tadj1 = adjacent_triangle_vertices(const_cast<TrianglePtr>(this), t->neighbors[1]->triangle);
        bool tadj2 = adjacent_triangle_vertices(const_cast<TrianglePtr>(this), t->neighbors[2]->triangle);

        if(!(tadj0 && tadj1 && tadj2)) {
            std::cout << "error, partial triangle neighbor triangle not adjacent to this triangle" << std::endl;
            return false;
        }
    }
    return true;

}


bool MxTriangle::isValid() const  {

    for(int cellId = 0; cellId < 2; ++cellId) {
        for(int adjId = 0; adjId < 3; ++adjId) {
            if(!partialTriangles[cellId].neighbors[adjId]) {
                std::cout << "error, triangle:" << this << std::endl
                        << ", partialTriangles["
                        << cellId << "].neighbors["
                        << adjId << "] is null"
                        << std::endl;
                return false;
            }

            if(!partialTriangles[cellId].neighbors[adjId]->triangle) {
                std::cout << "error, triangle:" << this << std::endl
                        << ", partialTriangles["
                        << cellId << "].neighbors["
                        << adjId << "]->triangle is null"
                        << std::endl;
                return false;
            }

            if(!adjacent_triangle_vertices(this, partialTriangles[cellId].neighbors[adjId]->triangle)) {
                std::cout << "error, triangle:" << this << std::endl
                        << ", partialTriangles["
                        << cellId << "].neighbors["
                        << adjId << "]->triangle does not have adjacent vertices to this triangle"
                        << std::endl;
                return false;
            }

            if(!incident(cells[cellId], partialTriangles[cellId].neighbors[adjId]->triangle)) {
                std::cout << "error, triangle:" << this << std::endl
                        << ", partialTriangles["
                        << cellId << "].neighbors["
                        << adjId << "]->triangle: "
                        << partialTriangles[cellId].neighbors[adjId]->triangle << std::endl
                        << "is not incident to cell[" << cellId << "]"
                        << std::endl
                        << "this: " << this << std::endl
                        << "neighbor: " << partialTriangles[cellId].neighbors[adjId]->triangle << std::endl;
                return false;
            }
        }
    }

    for(int i = 0; i < 3; ++i) {
        VertexPtr v1 = vertices[i];
        VertexPtr v2 = vertices[(i+1)%3];
        int ni = adjacentEdgeIndex(v1, v2);
        assert(ni == i);

        for(int j = 0; j < 2; ++j) {
            TrianglePtr tri = partialTriangles[j].neighbors[i]->triangle;
            if(tri->adjacentEdgeIndex(v1, v2) < 0) {
                std::cout << "error, triangle:" << this << std::endl
                        << ", neighbor triangle: " << tri << std::endl
                        << "at index[cell:" << j << ",nid:" << i
                        << "], neighbor triangle should be incident to vertices: " << std::endl
                        << "v1:" << v1 << ", v2: " << v2 << std::endl;
                return false;
            }
        }
    }

    for(int i = 0; i < 2; ++i) {
        if(!cells[i]) {
            std::cout << "error, triangle:" << this << std::endl
                    << ", cell[" << i << "] is null" << std::endl;
            return false;
        }

        if(!incident(const_cast<TrianglePtr>(this), cells[i])) {
            std::cout << "error, triangle:" << this << std::endl
                    << ", triangle is not incident to cell[" << i << "]" << std::endl;
            return false;
        }

        if(cells[i]->isRoot()) {
            assert(partialTriangles[i].mass == 0.);
        } else {
            isfinite(partialTriangles[i].mass ) && partialTriangles[i].mass  > 0;
        }
    }

    if(!isConnected()) {
        std::cout << "error, triangle:" << this << std::endl
                << " is not connected" << std::endl;
        return false;
    }

    if(!isfinite(area)) {
        std::cout << "error, triangle:" << this << std::endl
                << ", area is not finite" << std::endl;
        return false;
    }

    if(area < 0) {
        std::cout << "error, triangle:" << this << std::endl
                << ", area is negative" << std::endl;
        return false;
    }

    //if(!isfinite(aspectRatio)) {
    //    std::cout << "error, triangle:" << this << std::endl
    //            << ", aspect ratio is not finite" << std::endl;
    //    return false;
    //}

    //if(aspectRatio <= 0) {
    //    std::cout << "error, triangle:" << this << std::endl
    //            << ", aspect ratio is negative" << std::endl;
    //    return false;
    //}

    if(!isfinite(getMass())) {
        std::cout << "error, triangle:" << this << std::endl
                << ", mass is not finite" << std::endl;
        return false;
    }

    if(getMass() < 0) {
        std::cout << "error, triangle:" << this << std::endl
                << ", mass is negative" << std::endl;
        return false;
    }

    //if(!isfinite(normal.length())) {
    //    std::cout << "error, triangle:" << this << std::endl
    //            << ", normal.length() is not finite" << std::endl;
    //    return false;
    //}

    return true;
}

TrianglePtr MxTriangle::nextTriangleInFan(CVertexPtr vert,
                                          CCellPtr cell, CTrianglePtr prev) const {
    const MxPartialTriangle *pt = (cell == cells[0]) ? &partialTriangles[0] :
    (cell == cells[1]) ? &partialTriangles[1] : nullptr;

    if(!pt) return nullptr;

    // if we don't have a prev triangle, just grab the first
    // triangle we find.
    if(!prev) {
        for(uint i = 0; i < 3; ++i) {
            // the neighbor might be null
            if (pt->neighbors[i] && incident(pt->neighbors[i], vert)) {
                return pt->neighbors[i]->triangle;
            }
        }
    }
    else {
        const MxPartialTriangle *prevPt = (cell == prev->cells[0]) ? &prev->partialTriangles[0] :
        (cell == prev->cells[1]) ? &prev->partialTriangles[1] : nullptr;

        if(!prevPt) return nullptr;

        for(uint i = 0; i < 3; ++i) {
            if (pt->neighbors[i] && pt->neighbors[i] != prevPt && incident(pt->neighbors[i], vert)) {
                return pt->neighbors[i]->triangle;
            }
        }
    }
    return nullptr;
}

#ifndef NDEBUG
static TrianglePtr debugTriangleInRing(CTrianglePtr prev, CTrianglePtr curr)
{
    assert(prev);

    int cellIndx;

    int triId = -1;

    for(cellIndx = 0; cellIndx < 2; ++cellIndx) {
        for(int i = 0; triId < 0 && i < 3; ++i) {
            if(curr->partialTriangles[cellIndx].neighbors[i] &&
               curr->partialTriangles[cellIndx].neighbors[i]->triangle == prev) {
                triId = i;
                goto done;
            }
        }
    }
    
    done:
    
    if(triId < 0) {
        return nullptr;
    }
    
    assert(curr->cells[cellIndx] == prev->cells[0] || curr->cells[cellIndx] == prev->cells[1]);

    int oppoIndx = (cellIndx+1)%2;
    return curr->partialTriangles[oppoIndx].neighbors[triId]->triangle;
}
#endif

TrianglePtr MxTriangle::nextTriangleInRing(CTrianglePtr prev) const
{
    assert(prev);

    int cellIndx;

    //if(prev->cells[0] == cells[0] || prev->cells[1] == cells[0]) {
    //    cellIndx = 0;
    //}
    //else if(prev->cells[0] == cells[1] || prev->cells[1] == cells[1]) {
    //    cellIndx = 1;
    //}
    //else {
    //    return nullptr;
    //}

    int triId = -1;

    for(cellIndx = 0; cellIndx < 2; ++cellIndx) {
        for(int i = 0; triId < 0 && i < 3; ++i) {
            if(partialTriangles[cellIndx].neighbors[i] &&
               partialTriangles[cellIndx].neighbors[i]->triangle == prev) {
                triId = i;
                goto done;
            }
        }
    }
    
  done:

    if(triId < 0) {
        return nullptr;
    }
    
    assert(cells[cellIndx] == prev->cells[0] || cells[cellIndx] == prev->cells[1]);

    int oppoIndx = (cellIndx+1)%2;

#ifndef NDEBUG

    assert(partialTriangles[oppoIndx].neighbors[triId]);
    TrianglePtr next = partialTriangles[oppoIndx].neighbors[triId]->triangle;
    assert(next != this);
    assert(adjacent_triangle_vertices(prev, next));
    assert(adjacent_triangle_vertices(this, next));
    assert(debugTriangleInRing(next, this) == prev);

    return next;

#else
    return partialTriangles[oppoIndx].neighbors[triId]->triangle;
#endif


    assert(0 && "could not find other triangle in ring, partial face pointers probably wrong");
    return nullptr;
}

TrianglePtr MxTriangle::adjacentTriangleForEdge(CVertexPtr v1,
                                                CVertexPtr v2) const {
    for(int i = 0; i < 3; ++i) {
        assert(partialTriangles[0].neighbors[i]);
        TrianglePtr tri = partialTriangles[0].neighbors[i]->triangle;
        assert(tri);
        if(incident(tri, v1) && incident(tri, v2)) {
            return tri;
        }
    }
    return nullptr;
}

bool MxPartialTriangle::isValid() const
{
    assert(triangle);
    int id = (&triangle->partialTriangles[0] == this) ? 0 : 1;

    for(int adjId = 0; adjId < 3; ++adjId) {
        if(!neighbors[adjId]) {
            std::cout << "error, partial triangle id:"
                    << triangle->id << "." << id
                    << ", neighbors[" << adjId << "] is null"
                    << std::endl;
            return false;
        }

        if(!adjacent_triangle_vertices(triangle, neighbors[adjId]->triangle)) {
            std::cout << "error, partial triangle id:"
                    << triangle->id << "." << id
                    << ", neighbors[" << adjId << "]->triangle does not have adjacent vertices to this triangle"
                    << std::endl;
            return false;
        }

        if(!adjacent_triangle_pointers(triangle, neighbors[adjId]->triangle)) {
            std::cout << "error, partial triangle id:"
                    << triangle->id << "." << id
                    << ", neighbors[" << adjId << "]->triangle does not have adjacent pointers to this triangle"
                    << std::endl;
            return false;
        }
    }

    if(triangle->cells[id]->isRoot()) {
        if(mass != 0.) {
            std::cout << "error, partial triangle id:"
                    << triangle->id << "." << id
                    << ", mass for root cell partial triangle must be zero"
                    << std::endl;
            return false;
        }
    } else {
        if(!isfinite(mass)) {
            std::cout << "error, partial triangle id:"
                    << triangle->id << "." << id
                    << ", mass is not finite"
                    << std::endl;
            return false;
        }
    }

    return true;
}
