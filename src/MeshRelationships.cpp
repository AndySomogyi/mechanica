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

HRESULT disconnectPolygonCell(PolygonPtr poly, CellPtr cell)
{
    int cellIndex = poly->cellIndex(cell);

    if(cellIndex < 0) {
        return mx_error(E_INVALIDARG, "polygon is not connected to cell");
    }

    int polyIndex = indexOf(cell->surface, &poly->partialPolygons[cellIndex]);

    assert(polyIndex >= 0);

    cell->surface.erase(cell->surface.begin() + polyIndex);
    poly->cells[cellIndex] = nullptr;

    return S_OK;
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
            for(int j = 0; j < poly->edges.size(); ++j) {
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



HRESULT disconnectPolygonEdgeVertex(PolygonPtr poly, EdgePtr edge, CVertexPtr v, EdgePtr *e1, EdgePtr *e2)
{
    if(!poly || !edge || !v) {
        return mx_error(E_INVALIDARG, "null arguments");
    }

    if(poly->size() <= 3) {
        return mx_error(E_FAIL, "can't disconnect edge from polygon with less than four sides");
    }

    if(edge->vertices[0] != v && edge->vertices[1] != v) {
        return mx_error(E_INVALIDARG, "edge is not connected to vertex");
    }

    int vertIndex = indexOf(poly->vertices, v); // TODO inefficient

    if(vertIndex < 0) {
        return mx_error(E_INVALIDARG, "vertex is not connected to polygon");
    }

    // index of given edge
    int edgeIndex = indexOf(poly->edges, edge);
    int prevIndex = loopIndex(edgeIndex - 1, poly->edges.size());
    int nextIndex = loopIndex(edgeIndex + 1, poly->edges.size());

    if(edgeIndex < 0) {
        return mx_error(E_INVALIDARG, "edge is not connected to polygon");
    }

    *e1 = poly->edges[prevIndex];
    *e2 = poly->edges[nextIndex];


    // remove the edge / vertex from the poly
    poly->edges.erase(poly->edges.begin() + edgeIndex);
    poly->vertices.erase(poly->vertices.begin() + vertIndex);
    poly->_vertexAreas.erase(poly->_vertexAreas.begin() + vertIndex);
    poly->_vertexNormals.erase(poly->_vertexNormals.begin() + vertIndex);

    VERIFY(edge->erasePolygon(poly));


    return S_OK;
}

HRESULT insertPolygonEdge(PolygonPtr poly, EdgePtr edge)
{
    assert(poly->checkEdges());

    if(!poly || !edge ) {
        return mx_error(E_INVALIDARG, "null arguments");
    }

    if(!edge->vertices[0] || !edge->vertices[1]) {
        return mx_error(E_INVALIDARG, "one or more null vertices on edge");
    }

    // find the reference vertex;
    int refVertPolyIndex;
    int refVertEdgeIndex;

    {
        int tmp = indexOf(poly->vertices, edge->vertices[0]);
        if(tmp >= 0) {
            refVertPolyIndex = tmp;
            refVertEdgeIndex = 0;
        }
        else if((tmp = indexOf(poly->vertices, edge->vertices[1])) >= 0) {
            refVertPolyIndex = tmp;
            refVertEdgeIndex = 1;
        }
        else {
            return mx_error(E_INVALIDARG, "edge does not contain a vertex connected to polygon");
        }
    }

    VertexPtr newVert = edge->vertices[(refVertEdgeIndex + 1) % 2];

    // make sure other edge vertex is not is this poly already
    if(indexOf(poly->vertices, newVert) >= 0) {
        return mx_error(E_INVALIDARG, "both vertices of edge connected to poly");
    }

    // using std::vector::insert, if given begin() + size(), inserts at the end of the array.
    int nextPos = refVertPolyIndex+1;

    VERIFY(edge->insertPolygon(poly));

    // the given edge gets inserted at the found vertex pos, and all the
    // ones after it get pushed up. Find the next edge, and re-target it
    // to the new vertex. The new edge already connects the current ref vertex
    // and the new vertex.

    // push all the items down one, and insert the new values.
    poly->vertices.insert(poly->vertices.begin() + nextPos, newVert);
    poly->edges.insert(poly->edges.begin() + refVertPolyIndex, edge);
    poly->_vertexNormals.insert(poly->_vertexNormals.begin() + nextPos, Vector3{});
    poly->_vertexAreas.insert(poly->_vertexAreas.begin() + nextPos, 0);

    return S_OK;
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

bool connectedEdgeVertex(CEdgePtr edge, CVertexPtr v)
{
    return edge->vertices[0] == v || edge->vertices[1] == v;
}

HRESULT disconnectEdgeVertexFromPolygons(EdgePtr e, CVertexPtr v)
{
    return E_NOTIMPL;
}

HRESULT replacePolygonEdgeAndVerticesWithVertex(PolygonPtr poly, EdgePtr edge,
        VertexPtr newVert, EdgePtr* prevEdge, EdgePtr* nextEdge)
{
    if(!poly || !edge) {
        return mx_error(E_INVALIDARG, "null arguments" );
    }

    int index = indexOf(poly->edges, edge);

    if(index < 0) {
        return mx_error(E_INVALIDARG, "edge is not in polygon");
    }

    *prevEdge = wrappedAt(poly->edges, index-1);
    *nextEdge = wrappedAt(poly->edges, index+1);

    // remove the edge / vertex from the poly
    poly->edges.erase(poly->edges.begin() + index);
    poly->vertices.erase(poly->vertices.begin() + index);
    poly->_vertexAreas.erase(poly->_vertexAreas.begin() + index);
    poly->_vertexNormals.erase(poly->_vertexNormals.begin() + index);

    poly->vertices[loopIndex(index, poly->vertices.size())] = newVert;

    return S_OK;
}

HRESULT getPolygonAdjacentEdges(CPolygonPtr poly, CEdgePtr edge,
        EdgePtr* prevEdge, EdgePtr* nextEdge)
{
    if(!poly || !edge) {
        return mx_error(E_INVALIDARG, "null arguments" );
    }

    int index = indexOf(poly->edges, edge);
    if(index < 0) {
        return mx_error(E_INVALIDARG, "edge is not in polygon");
    }

    *prevEdge = wrappedAt(poly->edges, index-1);
    *nextEdge = wrappedAt(poly->edges, index+1);

    return S_OK;
}

HRESULT splitPolygonEdge(PolygonPtr poly, EdgePtr newEdge, EdgePtr refEdge)
{
    std::cout << "splitting polygon edge {" << std::endl;
    std::cout << "    poly: " << poly << std::endl;
    std::cout << "    newEdge: " << newEdge << std::endl;
    std::cout << "    refEdge: " << refEdge << std::endl;
    std::cout << "}" << std::endl;

    if(!poly || !refEdge || !newEdge) {
        return mx_error(E_INVALIDARG, "null arguments");
    }

    if(!newEdge->vertices[0] || !newEdge->vertices[1]) {
        return mx_error(E_INVALIDARG, "one or more null vertices on edge");
    }

    // find the reference vertex;
    int refVertPolyIndex;
    int refVertEdgeIndex;

    {
        // TODO: inefficient, only need to check against edge, but
        // check whole poly for now for consistency.
        int tmp = indexOf(poly->vertices, newEdge->vertices[0]);
        if(tmp >= 0) {
            refVertPolyIndex = tmp;
            refVertEdgeIndex = 0;
            if(!connectedEdgeVertex(refEdge, newEdge->vertices[0])) {
                return mx_error(E_INVALIDARG, "new edge is not connected to existing edge");
            }
        }
        else if((tmp = indexOf(poly->vertices, newEdge->vertices[1])) >= 0) {
            refVertPolyIndex = tmp;
            refVertEdgeIndex = 1;
            if(!connectedEdgeVertex(refEdge, newEdge->vertices[1])) {
                return mx_error(E_INVALIDARG, "new edge is not connected to existing edge");
            }
        }
        else {
            return mx_error(E_INVALIDARG, "new edge does not contain a vertex connected to polygon");
        }
    }

    VertexPtr newVert = newEdge->vertices[(refVertEdgeIndex + 1) % 2];

    // make sure other edge vertex is not is this poly already
    if(indexOf(poly->vertices, newVert) >= 0) {
        return mx_error(E_INVALIDARG, "both vertices of edge connected to poly");
    }

    int edgeIndex = indexOf(poly->edges, refEdge);

    if(edgeIndex < 0) {
        return mx_error(E_INVALIDARG, "reference edge not in polygon");
    }

    int edgeInsertPos, vertInsertPos;

    if(edgeIndex == refVertPolyIndex) {
        edgeInsertPos = refVertPolyIndex;
        vertInsertPos = refVertPolyIndex + 1;
    }
    else if(loopIndex(edgeIndex + 1, poly->edges.size()) == refVertPolyIndex) {
        edgeInsertPos = refVertPolyIndex;
        vertInsertPos = refVertPolyIndex;
    }
    else {
        return mx_error(E_INVALIDARG, "reference edge not in adjacent to new edge");
    }

    // using std::vector::insert, if given begin() + size(), inserts at the end of the array.

    VERIFY(newEdge->insertPolygon(poly));

    // the given edge gets inserted at the found vertex pos, and all the
    // ones after it get pushed up. Find the next edge, and re-target it
    // to the new vertex. The new edge already connects the current ref vertex
    // and the new vertex.

    // push all the items down one, and insert the new values.
    poly->vertices.insert(poly->vertices.begin() + vertInsertPos, newVert);
    poly->edges.insert(poly->edges.begin() + edgeInsertPos, newEdge);
    poly->_vertexNormals.insert(poly->_vertexNormals.begin() + vertInsertPos, Vector3{});
    poly->_vertexAreas.insert(poly->_vertexAreas.begin() + vertInsertPos, 0);

    std::cout << "updated polygon: " << poly << std::endl;

    return S_OK;
}

HRESULT replacePolygonVertexWithEdgeAndVertices(PolygonPtr poly, CVertexPtr vert,
        CEdgePtr e0, CEdgePtr e1,  EdgePtr edge, VertexPtr v0, VertexPtr v1) {
    int e0Index = poly->edgeIndex(e0);
    int e1Index = poly->edgeIndex(e1);

    std::cout << MX_FUNCTION << std::endl;
    std::cout << "poly: " << poly << std::endl;
    std::cout << "vert: " << vert << std::endl;
    std::cout << "e0: " << e0 << std::endl;
    std::cout << "e1: " << e1 << std::endl;
    std::cout << "edge: " << edge << std::endl;
    std::cout << "v0: " << v0 << std::endl;
    std::cout << "v1: " << v1 << std::endl;

    if(e0Index < 0 || e1Index < 0) {
        return mx_error(E_FAIL, "edges do not belong to polygon");
    }

    int vIndex = poly->vertexIndex(vert);
    if(vIndex < 0) {
        return mx_error(E_FAIL, "vertex does not belong to polygon");
    }

    if(!connectedEdgeVertex(e0, vert)) {
        return mx_error(E_FAIL, "edge e0 is not connected to original vertex");
    }

    if(!connectedEdgeVertex(e1, vert)) {
        return mx_error(E_FAIL, "edge e1 is not connected to original vertex");
    }

    if(e0Index < e1Index) {
        // Index wise, we have if e0 is before e1, i.e if index of e0 is i, we have:
        // e0[i]:v[i]:e1[i+1] -> e0[i]:v0[i]:edge[i+1]:v1[i+1]:e1[i+2]

        poly->vertices[vIndex] = v0;
        std::vector<VertexPtr>::iterator vi = poly->vertices.begin() + vIndex;
        if(vi != poly->vertices.end()) {
            vi++;
        }
        poly->vertices.insert(vi, v1);

        std::vector<EdgePtr>::iterator ei = poly->edges.begin() + e0Index;
        if(ei != poly->edges.end()) {
            ei++;
        }
        poly->edges.insert(ei, edge);

        poly->_vertexNormals.insert(poly->_vertexNormals.begin() + vIndex, Vector3{});
        poly->_vertexAreas.insert(poly->_vertexAreas.begin() + vIndex, 0);

        std::cout << "poly after insert: " << poly << std::endl;
        assert(poly->edgeIndex(edge) == e0Index + 1);
        assert(poly->vertexIndex(v0) == vIndex);
        assert(poly->vertexIndex(v1) == vIndex + 1);

        return S_OK;
    }
    else {
        // Index wise, we have if e1  before e0, i.e if index of e1 is i, we have:
        // e1[i]:v[i]:e0[i+1] -> e1[i]:v0[i]:edge[i+1]:v1[i+1]:e0[i+2]

        poly->vertices[vIndex] = v1;
        std::vector<VertexPtr>::iterator vi = poly->vertices.begin() + vIndex;
        if(vi != poly->vertices.end()) {
            vi++;
        }
        poly->vertices.insert(vi, v0);

        std::vector<EdgePtr>::iterator ei = poly->edges.begin() + e1Index;
        if(ei != poly->edges.end()) {
            ei++;
        }
        poly->edges.insert(ei, edge);

        poly->_vertexNormals.insert(poly->_vertexNormals.begin() + vIndex, Vector3{});
        poly->_vertexAreas.insert(poly->_vertexAreas.begin() + vIndex, 0);

        std::cout << "poly after insert: " << poly << std::endl;
        assert(poly->edgeIndex(edge) == e1Index + 1);
        assert(poly->vertexIndex(v1) == vIndex);
        assert(poly->vertexIndex(v0) == vIndex + 1);
        return S_OK;
    }


    return E_FAIL;
}
