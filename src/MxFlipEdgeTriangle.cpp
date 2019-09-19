/*
 * HITransition.cpp
 *
 *  Created on: Jan 16, 2019
 *      Author: andy
 */

#include "MeshOperations.h"
#include "MxDebug.h"

#define __fastcall
#include "DirectXMath.h"



bool Mx_IsTriangleToEdgeConfiguration(CEdgePtr edge) {
    return false;
}


/**
 * determine if the edge is a valid edge to triangle flip candidate.
 *
 * edgeCells[3]: enumerate the cells around the edge.
 * poly_2 : cell_0 : poly_0
 * poly_0 : cell_1 : poly_1
 * poly_1 : cell_2 : poly_2
 */
static bool isEdgeToTriangleConfiguration(CEdgePtr edge, CellPtr *edgeCells, CellPtr *endCells) {

    // make sure we have 2 vertices and 3 polys
    if(edge->vertexCount() != 2 || edge->polygonCount() != 3) {
        return false;
    }

    // make sure each poly has at least 4 sides
    for(uint i = 0; i < 3; ++i) {
        if(edge->polygons[i]->edges.size() < 4) {
            return false;
        }
    }

    // cells for 0 and 1 vertices.
    std::set<CellPtr> cells, cells0, cells1;

    // cells around the edge
    for(uint i = 0; i < 3; ++i) {
        cells.insert(edge->polygons[i]->cells[0]);
        cells.insert(edge->polygons[i]->cells[1]);
    }

    assert(cells.size() == 3);
    
    //for(CCellPtr c : cells) {
    //    std::cout << "cell around edge: " << c->id << std::endl;
    //}

    // grab the cells at the top and bottom of the edge
    for(uint i = 0; i < 3; ++i) {
        CPolygonPtr poly = edge->polygons[i];
        int edgeIndex = poly->edgeIndex(edge);
        int prevIndex = loopIndex(edgeIndex-1, poly->edges.size());
        int nextIndex = loopIndex(edgeIndex+1, poly->edges.size());

        CEdgePtr e = poly->edges[prevIndex];
        for(uint j = 0; j < e->polygonCount(); ++j) {
            CPolygonPtr p = e->polygons[j];
            
            //std::cout << "cells in polygon " << j << " for prev edge {" << p->cells[0]->id << ", " << p->cells[1]->id << "}" << std::endl;
            
            if(p->vertexIndex(edge->vertices[0]) >= 0) {
                if(cells.find(p->cells[0]) == cells.end()) {
                    cells0.insert(p->cells[0]);
                }
                if(cells.find(p->cells[1]) == cells.end()) {
                    cells0.insert(p->cells[1]);
                }
            }
            if(p->vertexIndex(edge->vertices[1]) >= 0) {
                if(cells.find(p->cells[0]) == cells.end()) {
                    cells1.insert(p->cells[0]);
                }

                if(cells.find(p->cells[1]) == cells.end()) {
                    cells1.insert(p->cells[1]);
                }
            }
        }
        
        e = poly->edges[nextIndex];
        for(uint j = 0; j < e->polygonCount(); ++j) {
            CPolygonPtr p = e->polygons[j];
            if(p->vertexIndex(edge->vertices[0]) >= 0) {
                if(cells.find(p->cells[0]) == cells.end()) {
                    cells0.insert(p->cells[0]);
                }
                if(cells.find(p->cells[1]) == cells.end()) {
                    cells0.insert(p->cells[1]);
                }
            }
            if(p->vertexIndex(edge->vertices[1]) >= 0) {
                if(cells.find(p->cells[0]) == cells.end()) {
                    cells1.insert(p->cells[0]);
                }
                
                if(cells.find(p->cells[1]) == cells.end()) {
                    cells1.insert(p->cells[1]);
                }
            }
        }
    }
    
    if(cells1.size() == 0 || cells0.size() == 0) {
        return false;
    }

    assert(cells1.size() == 1);
    assert(cells0.size() == 1);

    CellPtr cell0 = *cells0.begin();
    CellPtr cell1 = *cells1.begin();
    
    if(edgeCells) {
        for(int i = 0; i < 3; ++i) {
            PolygonPtr p = edge->polygons[i];
            PolygonPtr pp = edge->polygons[loopIndex(i+1, 3)];
            if(connectedPolygonCellPointers(p, pp->cells[0])) {
                edgeCells[i] = pp->cells[0];
            }
            else {
                assert(connectedPolygonCellPointers(p, pp->cells[1]));
                edgeCells[i] = pp->cells[1];
            }
        }
    }
    
    if(endCells) {
        endCells[0] = cell0;
        endCells[1] = cell1;
    }

    return (cell0 != cell1);
}

bool Mx_IsEdgeToTriangleConfiguration(CEdgePtr edge) {
    return isEdgeToTriangleConfiguration(edge, nullptr, nullptr);
}

HRESULT Mx_FlipTriangleToEdge(MeshPtr mesh, PolygonPtr poly, EdgePtr* edge)
{
    return E_NOTIMPL;
}


static HRESULT findUpperAndLowerEdgesForPolygon(CEdgePtr e, CPolygonPtr poly, EdgePtr *e0, EdgePtr *e1) {
    int edgeIndex = poly->edgeIndex(e);

    if(edgeIndex < 0) {
        return mx_error(E_FAIL, "polygon is not incident to edge");
    }
    
    int prevIndex = loopIndex(edgeIndex-1, poly->edges.size());
    int nextIndex = loopIndex(edgeIndex+1, poly->edges.size());

    EdgePtr ePrev = poly->edges[loopIndex(edgeIndex-1, poly->edges.size())];
    EdgePtr eNext = poly->edges[loopIndex(edgeIndex+1, poly->edges.size())];


    // if the edge is connected to the top vertex (0), its the top edge
    if(connectedEdgeVertex(ePrev, e->vertices[0])) {
        *e0 = ePrev;
        assert(!connectedEdgeVertex(ePrev, e->vertices[1]));
    }

    else if(connectedEdgeVertex(ePrev, e->vertices[1])) {
        *e1 = ePrev;
        assert(!connectedEdgeVertex(ePrev, e->vertices[0]));
    }
    else {
        return mx_error(E_FAIL, "previous edge is not connected to edge");
    }

    if(connectedEdgeVertex(eNext, e->vertices[0])) {
        *e0 = eNext;
        assert(!connectedEdgeVertex(eNext, e->vertices[1]));
    }

    else if(connectedEdgeVertex(eNext, e->vertices[1])) {
        *e1 = eNext;
        assert(!connectedEdgeVertex(eNext, e->vertices[0]));
    }
    
    assert(*e0 && *e1);

    return S_OK;
}

/**
 * Finds the polygon that is incident to both of the given edges.
 *
 * Based on proposition xxx, a pair of adjacent edges (i.e. pair of edges that
 * share a common vertex) have exactly one common polygon.
 */
static HRESULT findPolygonForEdges(CEdgePtr ePrev, CEdgePtr eNext, PolygonPtr *poly) {
    for(PolygonPtr p : ePrev->polygons) {
        if(eNext->polygonIndex(p) >= 0) {
            *poly = p;
            return S_OK;
        }
    }
    return mx_error(E_FAIL, "given edges do not share a polygon");
}

static VertexPtr otherVertex(EdgePtr e, VertexPtr v) {
    if(e->vertices[0] == v) {
        return e->vertices[1];
    }
    else {
        return e->vertices[0];
    }
}

HRESULT Mx_FlipEdgeToTriangle(MeshPtr mesh, EdgePtr edge, PolygonPtr* poly)
{
    CellPtr edgeCells[3] = {nullptr, nullptr, nullptr};
    CellPtr endCells[2] = {nullptr, nullptr};
    VertexPtr newVerts[3] = {nullptr, nullptr, nullptr};
    EdgePtr newEdges[3] = {nullptr, nullptr, nullptr};
    PolygonPtr upperPoly[3] = {nullptr, nullptr, nullptr};
    PolygonPtr lowerPoly[3] = {nullptr, nullptr, nullptr};
    EdgePtr upperEdges[3] = {nullptr, nullptr, nullptr};
    EdgePtr lowerEdges[3] = {nullptr, nullptr, nullptr};
    PolygonPtr newPoly = nullptr;
    HRESULT result = E_FAIL;

    if(!isEdgeToTriangleConfiguration(edge, edgeCells, endCells)) {
        return E_FAIL;
    }
    
    std::cout << MX_FUNCTION << std::endl << "edge: " << edge << std::endl;
    
    for(int i = 0; i < 3; ++i) {
        std::cout << "edge cells[" << i << "] : " << edgeCells[i] << std::endl;
    }
    
    for(int i = 0; i < 2; ++i) {
        std::cout << "end cells[" << i << "] : " << endCells[i] << std::endl;
    }
    
    for(int i = 0; i < 3; ++i) {
        std::cout << "edge polygon[" << i << "] : " << edge->polygons[i] << std::endl;
    }

    // grab the edges for each of the polygons, i.e. find all of the six upper and
    // lower edges
    for(int i = 0; i < 3; ++i) {
        if((result = findUpperAndLowerEdgesForPolygon(edge, edge->polygons[i],
                &upperEdges[i], &lowerEdges[i])) != S_OK) {
            return result;
        }
        std::cout << "upper edge[" << i << "]: " << upperEdges[i] << std::endl;
        std::cout << "lower edge[" << i << "]: " << lowerEdges[i] << std::endl;
    }

    // grab the upper and lower polygons for the radial cells
    for(int i = 0; i < 3; ++i) {
        if((result = findPolygonForEdges(upperEdges[i], upperEdges[loopIndex(i+1, 3)], &upperPoly[i])) != S_OK) {
            return result;
        }
        
        std::cout << "upper polygon[" << i << "]: " << upperPoly[i] << std::endl;
        
        assert(connectedCellPolygonPointers(edgeCells[i], upperPoly[i])
                && "found polygon is not connected to cell");
        assert(connectedCellPolygonPointers(endCells[0], upperPoly[i]) &&
                "upper polygon is not connected to upper cell");

    }

    for(int i = 0; i < 3; ++i) {
        if((result = findPolygonForEdges(lowerEdges[i], lowerEdges[loopIndex(i+1, 3)], &lowerPoly[i])) != S_OK) {
            return result;
        }
        
        std::cout << "lower polygon[" << i << "]: " << lowerPoly[i] << std::endl;
        
        assert(connectedCellPolygonPointers(edgeCells[i], lowerPoly[i])
                && "found polygon is not connected to cell");
        assert(connectedCellPolygonPointers(endCells[1], lowerPoly[i]) &&
                "lower polygon is not connected to lower cell");

    }

    // make the new vertices
    // create new  vertices in the plane of the radial polygon, at the average
    // position of the center of the edge and the opposite vertex of the
    // two connected edges.
    Vector3 centroid = (edge->vertices[0]->position + edge->vertices[1]->position) / 2.;
    for(int i = 0; i < 3; ++i) {
        Vector3 upPos = otherVertex(upperEdges[i], edge->vertices[0])->position;
        Vector3 lowPos = otherVertex(lowerEdges[i], edge->vertices[1])->position;
        Vector3 avgPos = (centroid + upPos + lowPos) / 3.;
        newVerts[i] = mesh->createVertex(avgPos);
    }

    // make new edges for the new triangle we'll create
    for(int i = 0; i < 3; ++i) {
        newEdges[i] = mesh->createEdge(MxEdge_Type, newVerts[i], newVerts[(i+1) % 3]);
        std::cout << "new edge[" << i << "] : " << newEdges[i] << std::endl;
    }

    // createPolygon finds the given vertices and edges, and hooks them up to the
    // new polygon
    newPoly = mesh->createPolygon(MxPolygon_Type, {newVerts[0], newVerts[1], newVerts[2]});
    assert(newPoly);
    std::cout << "new polygon: " << newPoly << std::endl;

    assert(connectedEdgeVertex(newPoly->edges[0], newVerts[0]));
    assert(connectedEdgeVertex(newPoly->edges[0], newVerts[1]));
    assert(connectedEdgeVertex(newPoly->edges[1], newVerts[1]));
    assert(connectedEdgeVertex(newPoly->edges[1], newVerts[2]));
    assert(connectedEdgeVertex(newPoly->edges[2], newVerts[2]));
    assert(connectedEdgeVertex(newPoly->edges[2], newVerts[0]));

    // remove the center edge from all of the radial polygons, and
    // replace the edge from the radial polygons with the
    // new triangle corner that we just made
    for(int i = 0; i < 3; ++i) {
        EdgePtr e0 = nullptr, e1 = nullptr;
        result = replacePolygonEdgeAndVerticesWithVertex(edge->polygons[i], edge,
                    newVerts[i], &e0, &e1);

        assert(SUCCEEDED(result));
        assert(e0 == upperEdges[i] || e0 == lowerEdges[i]);
        assert(e1 == upperEdges[i] || e1 == lowerEdges[i]);
        
        std::cout << "radial edge[" << i << "] after removing center vertex: " << edge->polygons[i] << std::endl;
    }
    
    for(int i = 0; i < 3; ++i) {
        std::cout << "upper poly[" << i << "] before replace: " << upperPoly[i];
        std::cout << "lower poly[" << i << "] before replace: " << lowerPoly[i];
    }

    // replace the single vertex in the upper and lower polygons with the
    // new edge that we made

    //In polygon up0: ue2:v0:ue0 -> ue2:vn2: ne0: vn0:ue0
    //In polygon up1: ue0:v0:ue1 -> ue0:vn0: ne1: vn1:ue1
    //In polygon up2: ue1:v0:ue2 -> ue1:vn1: ne2: vn2:ue2

    //Similarly, for the lower polygons, we have:
    //In polygon lp0: le2:v1:le0 -> le2:vn2: ne0: vn0:le0
    //In polygon lp1: le0:v1:le1 -> le0:vn0: ne1: vn1:le1
    //In polygon lp2: le1:v1:le2 -> le1:vn1: ne2: vn2:le2
    VERIFY(replacePolygonVertexWithEdgeAndVertices(upperPoly[0], edge->vertices[0],
            upperEdges[0], upperEdges[1],  newEdges[0], newVerts[0], newVerts[1]));
    VERIFY(replacePolygonVertexWithEdgeAndVertices(upperPoly[1], edge->vertices[0],
            upperEdges[1], upperEdges[2],  newEdges[1], newVerts[1], newVerts[2]));
    VERIFY(replacePolygonVertexWithEdgeAndVertices(upperPoly[2], edge->vertices[0],
            upperEdges[2], upperEdges[0],  newEdges[2], newVerts[2], newVerts[0]));

    VERIFY(replacePolygonVertexWithEdgeAndVertices(lowerPoly[0], edge->vertices[1],
            lowerEdges[0], lowerEdges[1],  newEdges[0], newVerts[0], newVerts[1]));
    VERIFY(replacePolygonVertexWithEdgeAndVertices(lowerPoly[1], edge->vertices[1],
            lowerEdges[1], lowerEdges[2],  newEdges[1], newVerts[1], newVerts[2]));
    VERIFY(replacePolygonVertexWithEdgeAndVertices(lowerPoly[2], edge->vertices[1],
            lowerEdges[2], lowerEdges[0],  newEdges[2], newVerts[2], newVerts[0]));
    
    // connect the new edges to the upper and lower polygons
    for(int i = 0; i < 3; ++i) {
        std::cout << "connecting new edge[" << i << "] to upper and lower polygons: " << newEdges[i] << std::endl;
        VERIFY(connectEdgePolygonPointers(newEdges[i], upperPoly[i]));
        VERIFY(connectEdgePolygonPointers(newEdges[i], lowerPoly[i]));
    }

    
    for(int i = 0; i < 3; ++i) {
        std::cout << "upper poly[" << i << "] after replace: " << upperPoly[i];
        std::cout << "lower poly[" << i << "] after replace: " << lowerPoly[i];
    }
    
    for(int i = 0; i < 3; ++i) {
        VERIFY(reconnectEdgeVertex(upperEdges[i], newVerts[i], edge->vertices[0]));
        VERIFY(reconnectEdgeVertex(lowerEdges[i], newVerts[i], edge->vertices[1]));
    }
    
    for(int i = 0; i < 3; ++i) {
        std::cout << "upper poly[" << i << "] after reconnect: " << upperPoly[i] << std::endl;
        std::cout << "lower poly[" << i << "] after reconnect: " << lowerPoly[i] << std::endl;
        std::cout << "radial poly[" << i << "] after reconnect: " << edge->polygons[i] << std::endl;
    }

    std::cout << "validating radial polygons..." << std::endl;
    for(int i = 0; i < 3; ++i) {
        if(!edge->polygons[i]->checkEdges()) {
            std::cout << "radial polygon [" << i << "] edge check failed: " << edge->polygons[i] << std::endl;
            assert(0);
        }
    }
    
    std::cout << "validating upper polygons..." << std::endl;
    for(int i = 0; i < 3; ++i) {
        if(!upperPoly[i]->checkEdges()) {
            std::cout << "upper polygon [" << i << "] edge check failed: " << upperPoly[i] << std::endl;
            assert(0);
        }
    }
    
    std::cout << "validating lower polygons..." << std::endl;
    for(int i = 0; i < 3; ++i) {
        if(!lowerPoly[i]->checkEdges()) {
            std::cout << "lower polygon [" << i << "] edge check failed: " << lowerPoly[i] << std::endl;
            assert(0);
        }
    }

    // we've defined the triangle vertices as {0,1,2}, so CCW winding means that the
    // normal vector points towards original vertex 1, i.e. cell 1, so add it first
    // to the top cell (0), then cell (1)
    VERIFY(connectPolygonCell(newPoly, endCells[0]));
    VERIFY(connectPolygonCell(newPoly, endCells[1]));
    VERIFY(endCells[0]->topologyChanged());
    VERIFY(endCells[1]->topologyChanged());

    VERIFY(mesh->positionsChanged());

    return S_OK;
}
