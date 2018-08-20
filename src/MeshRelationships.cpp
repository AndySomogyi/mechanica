/*
 * MeshRelationships.cpp
 *
 *  Created on: Sep 29, 2017
 *      Author: andy
 */

#include <MeshRelationships.h>
#include <MxEdge.h>
#include "MxMesh.h"
#include <algorithm>
#include <iostream>

bool connectedPolygonCellPointers(CPolygonPtr t, CCellPtr c) {
    return t->cells[0] == c || t->cells[1] == c;
}

bool adjacentPolygonVertices(CPolygonPtr a, CPolygonPtr b) {


    return false;
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

HRESULT connectPolygonPolygonPointers(PolygonPtr a, PolygonPtr b)
{

    return E_FAIL;
}

bool incidentEdgePolygonVertices(CEdgePtr edge, CPolygonPtr tri)
{
    return incidentPolygonVertex(tri, edge->vertices[0]) && incidentPolygonVertex(tri, edge->vertices[1]);
}

bool connectedEdgePolygonPointers(CEdgePtr edge, CPolygonPtr poly)
{
    for(int i = 0; i < 3; ++i) {
        if(edge->polygons[i] == poly) {
#ifndef NDEBUG
            for(int j = 0; j > poly->edges.size(); ++j) {
                if(poly->edges[j] == edge) {
                    return true;
                }
            }
            assert(false && "edge connected to poly, but poly is not connected to edge");
#endif
            return true;
        }
    }
    return false;
}

HRESULT disconnectPolygonEdge(PolygonPtr poly, EdgePtr edge)
{
    if(poly->sides() <= 3) {
        return mx_error(E_FAIL, "can't disconnect edge from polygon with less than four sides");
    }





}

HRESULT insertEdgeVertexIntoPolygon(EdgePtr edge, VertexPtr vert,
        PolygonPtr poly, CVertexPtr ref)
{
}

HRESULT connectPolygonVertices(MeshPtr mesh, PolygonPtr poly,
        const std::vector<VertexPtr>& vertices)
{
    if(poly->vertices.size() != 0) {
        return mx_error(E_FAIL, "only empty polygons supported for now");
    }

    if(vertices.size() < 3) {
        return mx_error(E_FAIL, "only support polygons with at least three vertices");
    }

    // grab the edges for each vertex.
    poly->edges.resize(vertices.size(), nullptr);
    for(int i = 0; i < vertices.size(); ++i) {
        VertexPtr v1 = vertices[i];
        VertexPtr v2 = vertices[(i+1)%vertices.size()];
        EdgePtr edge = mesh->findEdge(v1, v2);

        if(edge == nullptr) {
            return mx_error(E_FAIL, "could not find edge for vertex");
        }

        if(connectedEdgePolygonPointers(edge, poly)) {
            return mx_error(E_FAIL, "edge is already connected to polygon");
        }

        uint pc = edge->polygonCount();

        if(pc >= 3) {
            return mx_error(E_FAIL, "edge is already connected to three polygons");
        }

        edge->polygons[pc] = poly;
        poly->edges[i] = edge;
    }


    poly->vertices = vertices;
    poly->_vertexNormals.resize(vertices.size());
    poly->_vertexAreas.resize(vertices.size());

    poly->positionsChanged();

    return S_OK;
}
