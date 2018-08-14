/*
 * MeshRelationships.cpp
 *
 *  Created on: Sep 29, 2017
 *      Author: andy
 */

#include <MeshRelationships.h>
#include <MxEdge.h>
#include <algorithm>
#include <iostream>

bool connectedPolygonCellPointers(CPolygonPtr t, CCellPtr c) {
    return t->cells[0] == c || t->cells[1] == c;
}

bool adjacentPolygonVertices(CPolygonPtr a, CPolygonPtr b) {


    return false;
}



typedef std::array<int, 2> EdgeIndx;

inline EdgeIndx adjacent_edge_indx(CPPolygonPtr a, CPPolygonPtr b) {
    assert(a->polygon != b->polygon && "partial triangles are on the same triangle");

    EdgeIndx result;

    VertexPtr v1 = nullptr, v2 = nullptr;


    for(int i = 0; i < 3  && (v1 == nullptr || v2 == nullptr); ++i) {
        for(int j = 0; j < 3 && (v1 == nullptr || v2 == nullptr); ++j) {
            if(!v1 && a->polygon->vertices[i] == b->polygon->vertices[j]) {
                v1 = a->polygon->vertices[i];
                continue;
            }
            if(!v2 && a->polygon->vertices[i] == b->polygon->vertices[j]) {
                v2 = a->polygon->vertices[i];
            }
        }
    }

    assert(v1 && v2);
    assert(v1 != v2);

    result[0] = a->polygon->adjacentEdgeIndex(v1, v2);
    result[1] = b->polygon->adjacentEdgeIndex(v1, v2);

    return result;
}

inline EdgeIndx adjacentEdgeIndex(CPolygonPtr a, CPolygonPtr b) {
    assert(a && b && a != b && "partial triangles are on the same triangle");

    EdgeIndx result = {-1, -1};

    VertexPtr v1 = nullptr, v2 = nullptr;


    for(int i = 0; i < 3  && (v1 == nullptr || v2 == nullptr); ++i) {
        for(int j = 0; j < 3 && (v1 == nullptr || v2 == nullptr); ++j) {
            if(!v1 && a->vertices[i] == b->vertices[j]) {
                v1 = a->vertices[i];
                continue;
            }
            if(!v2 && a->vertices[i] == b->vertices[j]) {
                v2 = a->vertices[i];
            }
        }
    }

    if(v1 || v2) {
        assert(v1 && v2);
        assert(v1 != v2);
        result[0] = a->adjacentEdgeIndex(v1, v2);
        result[1] = b->adjacentEdgeIndex(v1, v2);
    }

    return result;
}





bool incidentPolygonVertex(CPolygonPtr tri, CVertexPtr v) {
    assert(tri);
    return tri->vertices[0] == v || tri->vertices[1] == v || tri->vertices[2] == v;
}






bool connectedPolygonPointers(CPolygonPtr a, CPolygonPtr b)
{
    return true;
}

HRESULT connectPolygonCell(PolygonPtr tri, CellPtr cell, int index) {
#ifndef NDEBUG
    assert(index == 0 || index == 1);
    if(cell) assert(!tri->cells[index]);
#endif
    tri->cells[index] = cell;

    return S_OK;
}

HRESULT connectPolygonCell(PolygonPtr poly, CellPtr cell)
{
    if(poly->cells[0] && poly->cells[1]) {
        return mx_error(E_FAIL, "polygon does not have any empty sides");
    }

    int side = poly->cells[0] == nullptr ? 0 : 1;

    for(CPPolygonPtr p : cell->surface) {
        if(p->polygon == poly) {
            return mx_error(E_FAIL, "polygon already connected to cell");
        }
    }

    cell->surface.push_back(&poly->partialPolygons[side]);
    poly->cells[side] = cell;

    return S_OK;
}

HRESULT disconnectPolygonCell(PolygonPtr tri, CellPtr cell)
{
    return E_NOTIMPL;
}

HRESULT connectPolygonPolygon(PolygonPtr a, PolygonPtr b)
{

    return E_FAIL;
}

bool incidentEdgePolygonVertices(CSkeletalEdgePtr edge, CPolygonPtr tri)
{
    return incidentPolygonVertex(tri, edge->vertices[0]) && incidentPolygonVertex(tri, edge->vertices[1]);
}

bool connectedEdgePolygonPointers(CSkeletalEdgePtr edge, CPolygonPtr tri)
{
    for(int i = 0; i < 3; ++i) {
        if(tri->neighbors[i] == edge) {
#ifndef NDEBUG
            assert(0 && "edge triangle list does not contain triangle");
            int index = indexOfEdgeVertices(edge, tri);
            assert(index >= 0);
            assert(tri->neighbors[index] == edge);
            assert(edge->triangles[0] == tri || edge->triangles[1] == tri || edge->triangles[2] == tri);
#endif
            return true;
        }
    }
    return false;
}

int indexOfEdgeVertices(CSkeletalEdgePtr edge, CPolygonPtr tri)
{
    return tri->adjacentEdgeIndex(edge->vertices[0], edge->vertices[1]);
}
