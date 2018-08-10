/*
 * MxTriangle.cpp
 *
 *  Created on: Oct 3, 2017
 *      Author: andy
 */

#include <MxPolygon.h>
#include "MxCell.h"
#include "MxDebug.h"
#include <iostream>

static std::string to_string(CCellPtr cell) {
    return cell ? std::to_string(cell->id) : "null";
}

static std::string to_string(const MxPartialPolygon *pt) {
    return "triId:" + (pt ? std::to_string(pt->polygon->id) : "null");
}

std::ostream& operator<<(std::ostream& os, CPolygonPtr tri)
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


MxPolygon::MxPolygon(uint _id, MxPolygonType* type,
        const std::vector<VertexPtr>& verts,
        const std::array<CellPtr, 2>& cells,
        const std::array<MxPartialPolygonType*, 2>& partTriTypes) :
            id{_id}, MxObject{type}, vertices{verts}, cells{cells},
            partialTriangles{{{partTriTypes[0], this}, {partTriTypes[1], this}}} {

    neighbors.resize(vertices.size());
    partialTriangles[0].force.resize(vertices.size());
    partialTriangles[1].force.resize(vertices.size());


    // connect this triangle to the vertex tri lists
    for(VertexPtr vert : verts) {
        auto res = vert->appendTriangle(this);
        assert(res==S_OK);
    }

    positionsChanged();
}

/**
 * Neighbor triangle indexes are related to vertex indexes as
 * the i'th neighbor triangle shares vertices at indexes i and either i+1
 * or i-1.
 */
int MxPolygon::adjacentEdgeIndex(CVertexPtr a, CVertexPtr b) const {
    for(int i = 0; i < 3; ++i) {
        if((vertices[i] == a && vertices[(i+1)%3] == b) ||
           (vertices[i] == b && vertices[(i+1)%3] == a)) {
            return i;
        }
    }
    return -1;
}

HRESULT MxPolygon::positionsChanged() {

    centroid = {{0.f, 0.f, 0.f}};

    for (CVertexPtr v : vertices) {
        centroid += v->position;
    }

    centroid /= (float)vertices.size();

    return S_OK;
}


bool MxPolygon::isConnected() const {

    for(int i = 0; i < 2; ++i) {

        if(cells[i]->isRoot()) continue;

        // TODO: HACK fix the constness.

        const PPolygonPtr t = const_cast<PPolygonPtr>(&partialTriangles[i]);

        bool padj0 = adjacentPartialTrianglePointers(t, t->neighbors[0]);
        bool padj1 = adjacentPartialTrianglePointers(t, t->neighbors[1]);
        bool padj2 = adjacentPartialTrianglePointers(t, t->neighbors[2]);
        bool padj = padj0 && padj1 && padj2;

        // check pointers
        if (!padj) {
            std::cout << "error, partial triangles neighbors not adjacent to this triangle" << std::endl;
            return false;
        }

        assert(this == t->polygon);

        // check vertices
        bool tadj0 = adjacentTriangleVertices(const_cast<PolygonPtr>(this), t->neighbors[0]->polygon);
        bool tadj1 = adjacentTriangleVertices(const_cast<PolygonPtr>(this), t->neighbors[1]->polygon);
        bool tadj2 = adjacentTriangleVertices(const_cast<PolygonPtr>(this), t->neighbors[2]->polygon);

        if(!(tadj0 && tadj1 && tadj2)) {
            std::cout << "error, partial triangle neighbor triangle not adjacent to this triangle" << std::endl;
            return false;
        }
    }
    return true;

}


bool MxPolygon::isValid() const  {

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

            if(!partialTriangles[cellId].neighbors[adjId]->polygon) {
                std::cout << "error, triangle:" << this << std::endl
                        << ", partialTriangles["
                        << cellId << "].neighbors["
                        << adjId << "]->triangle is null"
                        << std::endl;
                return false;
            }

            if(!adjacentTriangleVertices(this, partialTriangles[cellId].neighbors[adjId]->polygon)) {
                std::cout << "error, triangle:" << this << std::endl
                        << ", partialTriangles["
                        << cellId << "].neighbors["
                        << adjId << "]->triangle does not have adjacent vertices to this triangle" << std::endl
                        << "offending neighbor: " << partialTriangles[cellId].neighbors[adjId]->polygon << std::endl;
                return false;
            }

            if(!connectedCellTrianglePointers(cells[cellId], partialTriangles[cellId].neighbors[adjId]->polygon)) {
                std::cout << "error, triangle:" << this << std::endl
                        << ", partialTriangles["
                        << cellId << "].neighbors["
                        << adjId << "]->triangle: "
                        << partialTriangles[cellId].neighbors[adjId]->polygon << std::endl
                        << "is not incident to cell[" << cellId << "]"
                        << std::endl
                        << "this: " << this << std::endl
                        << "neighbor: " << partialTriangles[cellId].neighbors[adjId]->polygon << std::endl;
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
            PolygonPtr tri = partialTriangles[j].neighbors[i]->polygon;
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

        if(!connectedTriangleCellPointers(const_cast<PolygonPtr>(this), cells[i])) {
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






PolygonPtr MxPolygon::adjacentTriangleForEdge(CVertexPtr v1,
                                                CVertexPtr v2) const {
    for(int i = 0; i < 3; ++i) {
        assert(partialTriangles[0].neighbors[i]);
        PolygonPtr tri = partialTriangles[0].neighbors[i]->polygon;
        assert(tri);
        if(incidentTriangleVertex(tri, v1) && incidentTriangleVertex(tri, v2)) {
            return tri;
        }
    }
    return nullptr;
}

bool MxPartialPolygon::isValid() const
{
    assert(polygon);
    int id = (&polygon->partialTriangles[0] == this) ? 0 : 1;

    for(int adjId = 0; adjId < 3; ++adjId) {
        if(!neighbors[adjId]) {
            std::cout << "error, partial triangle id:"
                    << polygon->id << "." << id
                    << ", neighbors[" << adjId << "] is null"
                    << std::endl;
            return false;
        }

        if(!adjacentTriangleVertices(polygon, neighbors[adjId]->polygon)) {
            std::cout << "error, partial triangle id:"
                    << polygon->id << "." << id
                    << ", neighbors[" << adjId << "]->triangle does not have adjacent vertices to this triangle, " << std::endl
                    << "offending neighbor: " << neighbors[adjId]->polygon << std::endl;
            return false;
        }

        if(!connectedTrianglePointers(polygon, neighbors[adjId]->polygon)) {
            std::cout << "error, partial triangle id:"
                    << polygon->id << "." << id
                    << ", neighbors[" << adjId << "]->triangle does not have adjacent pointers to this triangle"
                    << std::endl;
            return false;
        }
    }

    if(polygon->cells[id]->isRoot()) {
        if(mass != 0.) {
            std::cout << "error, partial triangle id:"
                    << polygon->id << "." << id
                    << ", mass for root cell partial triangle must be zero"
                    << std::endl;
            return false;
        }
    } else {
        if(!isfinite(mass)) {
            std::cout << "error, partial triangle id:"
                    << polygon->id << "." << id
                    << ", mass is not finite"
                    << std::endl;
            return false;
        }
    }

    return true;
}


Orientation MxPolygon::orientation() const
{
    Orientation o0{Orientation::Invalid}, o1{Orientation::Invalid};

    if(cells[0] && !cells[0]->isRoot()) {
        Vector3 triPos = centroid - cells[0]->centroid;
        float dir = Math::dot(normal, triPos);
        o0 = dir > 0 ? Orientation::Outward : Orientation::Inward;
    }

    if(cells[1] && !cells[1]->isRoot()) {
        Vector3 triPos = centroid - cells[1]->centroid;
        float dir = -1 * Math::dot(normal, triPos);
        o1 = dir > 0 ? Orientation::Outward : Orientation::Inward;
    }

    if((o0 == Orientation::Outward || o0 == Orientation::Invalid) &&
       (o1 == Orientation::Outward || o1 == Orientation::Invalid)) {
        return Orientation::Outward;
    }

    if((o0 == Orientation::Inward || o0 == Orientation::Invalid) &&
       (o1 == Orientation::Inward || o1 == Orientation::Invalid)) {
        return Orientation::Inward;
    }

    if((o0 == Orientation::Inward  || o0 == Orientation::Invalid) &&
       (o1 == Orientation::Outward || o1 == Orientation::Invalid)) {
        return Orientation::InwardOutward;
    }

    if((o0 == Orientation::Outward || o0 == Orientation::Invalid) &&
       (o1 == Orientation::Inward  || o1 == Orientation::Invalid)) {
        return Orientation::OutwardInward;
    }

    return Orientation::Invalid;
}
