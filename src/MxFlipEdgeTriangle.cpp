/*
 * HITransition.cpp
 *
 *  Created on: Jan 16, 2019
 *      Author: andy
 */

#include "MeshOperations.h"

#define __fastcall
#include "DirectXMath.h"



bool Mx_IsTriangleToEdgeConfiguration(CEdgePtr edge) {
    return false;
}

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
        int prevIndex = mod(edgeIndex-1, poly->edges.size());
        int nextIndex = mod(edgeIndex+1, poly->edges.size());

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
        int i = 0;
        for(CellPtr c : cells) {
            edgeCells[i++] = c;
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


/**
 * Really similar to a T1 flip in a surface, but generalized to a flip
 * in 3D where the edge, polys are one corner of a cell
 *
 * There are three cells around the edge, and cell is one of them.
 *
 *
 * Looking at the edge, from the side, from inside the given cell, we label the
 * polygons as:
 *
 *  \        /
 *   \  p1  /
 *    \    /
 *     \  /
 *      \/
 *      |
 *      |
 * p4   |   p2
 *      |
 *      |
 *     / \
 *    /   \
 *   /     \
 *  /   p3  \
 * /         \
 *
 *
 * @param cell: the cell corner that this acts on, cell is incident to edge.
 */
static HRESULT flipEdgeForCellCorner(MeshPtr mesh, CellPtr cell, EdgePtr edge) {

    std::cout << "applyT1Edge2Transition(edge=" << edge << ")" << std::endl;

    if(edge->polygonCount() != 3) {
        return mx_error(E_FAIL, "edge polygon count must be 3");
    }

    if(edge->polygons[0]->size() <= 3 || edge->polygons[1]->size() <= 3) {
        return mx_error(E_FAIL, "can't collapse edge that's connected to polygons with less than 3 sides");

    }

    PolygonPtr p1 = nullptr, p2 = nullptr, p3 = nullptr, p4 = nullptr;

    // top and bottom vertices
    VertexPtr v1 = edge->vertices[0];
    VertexPtr v2 = edge->vertices[1];

    // find polygon indices for pair of polygons that belong to the given cell
    int polyIndex[2] = {-1, -1};
    if(connectedPolygonCellPointers(edge->polygons[0], cell) &&
            connectedPolygonCellPointers(edge->polygons[1], cell)) {
        polyIndex[0] = 0;
        polyIndex[1] = 1;
    }
    else if(connectedPolygonCellPointers(edge->polygons[1], cell) &&
            connectedPolygonCellPointers(edge->polygons[2], cell)) {
        polyIndex[0] = 1;
        polyIndex[1] = 2;
    }
    else if(connectedPolygonCellPointers(edge->polygons[2], cell) &&
            connectedPolygonCellPointers(edge->polygons[0], cell)) {
        polyIndex[0] = 2;
        polyIndex[1] = 0;
    }
    else {
        assert("edge is not connected to cell" && 0);
    }


    // identify the two side polygons as p4 on the left, and p2 on the right, check
    // vertex winding. Choose p2 to have v2 after v1, and p4 to have v1 after v2
    {
        // TODO use next vert instead of looking at all vertices.
        int v1Index_0 = edge->polygons[polyIndex[0]]->vertexIndex(v1);
        int v2Index_0 = edge->polygons[polyIndex[0]]->vertexIndex(v2);

#ifndef NDEBUG
        int v1Index_1 = edge->polygons[polyIndex[1]]->vertexIndex(v1);
        int v2Index_1 = edge->polygons[polyIndex[1]]->vertexIndex(v2);
        assert(v1Index_0 >= 0 && v2Index_0 >= 0 && v1Index_0 != v2Index_0);
#endif

        if(((v1Index_0 + 1) % edge->polygons[0]->size()) == v2Index_0) {
            // found p2 CCW
            p2 = edge->polygons[polyIndex[0]];
            p4 = edge->polygons[polyIndex[1]];
            assert((v2Index_1 + 1) % edge->polygons[1]->size() == v1Index_1);
        }
        else {
            // found p4
            p4 = edge->polygons[polyIndex[0]];
            p2 = edge->polygons[polyIndex[1]];
            assert((v1Index_1 + 1) % edge->polygons[1]->size() == v2Index_1);
        }
    }

    assert(p4 && p2);

    EdgePtr e1 = nullptr, e2 = nullptr, e3 = nullptr, e4 = nullptr;


    std::cout << "poly p2: " << p2 << std::endl;
    std::cout << "poly p4: " << p4 << std::endl;

    std::cout << "disconnectPolygonEdgeVertex(p2, edge, v1, &e1, &e2)" << std::endl;
    VERIFY(disconnectPolygonEdgeVertex(p2, edge, v1, &e1, &e2));


    std::cout << "poly p2: " << p2 << std::endl;
    std::cout << "poly p4: " << p4 << std::endl;

    std::cout << "disconnectPolygonEdgeVertex(p4, edge, v2, &e3, &e4)" << std::endl;
    VERIFY(disconnectPolygonEdgeVertex(p4, edge, v2, &e3, &e4));

    //assert(edge->polygonCount() == 1);

    std::cout << "e1:" << e1 << std::endl;
    std::cout << "e2:" << e2 << std::endl;
    std::cout << "e3:" << e3 << std::endl;
    std::cout << "e4:" << e4 << std::endl;

    std::cout << "poly p2: " << p2 << std::endl;
    std::cout << "poly p4: " << p4 << std::endl;

    assert(connectedEdgeVertex(e1, v1));
    assert(connectedEdgeVertex(e2, v2));
    assert(connectedEdgeVertex(e3, v2));
    assert(connectedEdgeVertex(e4, v1));

    for(PolygonPtr p : e1->polygons) {
        if(contains(p->edges, e4)) {
            p1 = p;
            break;
        }
    }

    for(PolygonPtr p : e2->polygons) {
        if(contains(p->edges, e3)) {
            p3 = p;
            break;
        }
    }

    assert(p1 && p3);
    assert(p1 != p2 && p1 != p3 && p1 != p4);
    assert(p2 != p1 && p2 != p3 && p2 != p4);
    assert(p3 != p1 && p3 != p2 && p3 != p4);
    assert(p4 != p1 && p4 != p2 && p1 != p3);

    // original edge vector.
    Vector3 edgeVec = v1->position - v2->position;
    float halfLen = edgeVec.length() / 2;

    // center position of the polygons that will get a new edge connecting them.
    Vector3 centroid = (p2->centroid + p4->centroid) / 2;

    v2->position = centroid + (p2->centroid - centroid).normalized() * halfLen;
    v1->position = centroid + (p4->centroid - centroid).normalized() * halfLen;

    std::cout << "poly p1: " << p1 << std::endl;
    std::cout << "poly p2: " << p2 << std::endl;
    std::cout << "poly p3: " << p3 << std::endl;
    std::cout << "poly p4: " << p4 << std::endl;

    std::cout << "insertPolygonEdge(p1, edge)" << std::endl;
    VERIFY(insertPolygonEdge(p1, edge));

    std::cout << "poly p1: " << p1 << std::endl;
    std::cout << "poly p2: " << p2 << std::endl;
    std::cout << "poly p3: " << p3 << std::endl;
    std::cout << "poly p4: " << p4 << std::endl;

    std::cout << "insertPolygonEdge(p3, edge)" << std::endl;
    VERIFY(insertPolygonEdge(p3, edge));

    std::cout << "poly p1: " << p1 << std::endl;
    std::cout << "poly p2: " << p2 << std::endl;
    std::cout << "poly p3: " << p3 << std::endl;
    std::cout << "poly p4: " << p4 << std::endl;

    assert(connectedEdgeVertex(e1, v1));
    assert(connectedEdgeVertex(e2, v2));
    assert(connectedEdgeVertex(e3, v2));
    assert(connectedEdgeVertex(e4, v1));

    std::cout << "reconnecting edge vertices..." << std::endl;

    // reconnect the two diagonal edges, the other two edges, e2 and e4 stay
    // connected to their same vertices.
    VERIFY(reconnectEdgeVertex(e1, v2, v1));
    VERIFY(reconnectEdgeVertex(e3, v1, v2));

    std::cout << "poly p1: " << p1 << std::endl;
    std::cout << "poly p2: " << p2 << std::endl;
    std::cout << "poly p3: " << p3 << std::endl;
    std::cout << "poly p4: " << p4 << std::endl;

    assert(p1->size() >= 0);
    assert(p2->size() >= 0);
    assert(p3->size() >= 0);
    assert(p4->size() >= 0);

    assert(p1->checkEdges());
    assert(p2->checkEdges());
    assert(p3->checkEdges());
    assert(p4->checkEdges());

    for(CellPtr cell : mesh->cells) {
        cell->topologyChanged();
    }

    mesh->setPositions(0, 0);

    VERIFY(mesh->positionsChanged());

    return S_OK;
}

static HRESULT findUpperAndLowerEdgesForPolygon(CEdgePtr e, CPolygonPtr poly, EdgePtr *e0, EdgePtr *e1) {
    int edgeIndex = poly->edgeIndex(e);

    if(edgeIndex < 0) {
        return mx_error(E_FAIL, "polygon is not incident to edge");
    }

    EdgePtr ePrev = poly->edges[mod(edgeIndex-1, poly->edges.size())];
    EdgePtr eNext = poly->edges[mod(edgeIndex+1, poly->edges.size())];


    // if the edge is connected to the top vertex (0), its the top edge
    if(connectedEdgeVertex(ePrev, e->vertices[0])) {
        *e0 = ePrev;
        assert(!connectedEdgeVertex(ePrev, e->vertices[1]));
    }

    else if(connectedEdgeVertex(ePrev, e->vertices[1])) {
        *e1 = ePrev;
        assert(!connectedEdgeVertex(ePrev, e->vertices[0]));
    }

    if(connectedEdgeVertex(eNext, e->vertices[0])) {
        *e0 = eNext;
        assert(!connectedEdgeVertex(eNext, e->vertices[1]));
    }

    else if(connectedEdgeVertex(eNext, e->vertices[1])) {
        *e1 = eNext;
        assert(!connectedEdgeVertex(eNext, e->vertices[0]));
    }

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

    // grab the edges for each of the polygons, i.e. find all of the six upper and
    // lower edges
    for(int i = 0; i < 3; ++i) {
        if((result = findUpperAndLowerEdgesForPolygon(edge, edge->polygons[i],
                &upperEdges[i], &lowerEdges[i])) != S_OK) {
            return result;
        }
    }

    // grab the upper and lower polygons for the radial cells
    for(int i = 0; i < 3; ++i) {
        if((result = findPolygonForEdges(upperEdges[i], upperEdges[mod(i+1, 3)], &upperPoly[i])) != S_OK) {
            return result;
        }
        assert(connectedCellPolygonPointers(edgeCells[i], upperPoly[i])
                && "found polygon is not connected to cell");
        assert(connectedCellPolygonPointers(endCells[0], upperPoly[i]) &&
                "upper polygon is not connected to upper cell");

    }

    for(int i = 0; i < 3; ++i) {
        if((result = findPolygonForEdges(lowerEdges[i], lowerEdges[mod(i+1, 3)], &lowerPoly[i])) != S_OK) {
            return result;
        }
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
    }

    // createPolygon finds the given vertices and edges, and hooks them up to the
    // new polygon
    newPoly = mesh->createPolygon(MxPolygon_Type, {newVerts[0], newVerts[1], newVerts[2]});
    assert(newPoly);

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
            upperEdges[2], upperEdges[0],  newEdges[0], newVerts[2], newVerts[0]));
    VERIFY(replacePolygonVertexWithEdgeAndVertices(upperPoly[1], edge->vertices[0],
            upperEdges[0], upperEdges[1],  newEdges[1], newVerts[1], newVerts[1]));
    VERIFY(replacePolygonVertexWithEdgeAndVertices(upperPoly[2], edge->vertices[0],
            upperEdges[1], upperEdges[2],  newEdges[2], newVerts[1], newVerts[2]));

    VERIFY(replacePolygonVertexWithEdgeAndVertices(lowerPoly[0], edge->vertices[1],
            lowerEdges[2], lowerEdges[0],  newEdges[0], newVerts[2], newVerts[0]));
    VERIFY(replacePolygonVertexWithEdgeAndVertices(lowerPoly[1], edge->vertices[1],
            lowerEdges[0], lowerEdges[1],  newEdges[1], newVerts[0], newVerts[1]));
    VERIFY(replacePolygonVertexWithEdgeAndVertices(lowerPoly[2], edge->vertices[1],
            lowerEdges[1], lowerEdges[2],  newEdges[2], newVerts[1], newVerts[2]));




    return S_OK;
}
