/*
 * T2Transition.cpp
 *
 *  Created on: Aug 24, 2018
 *      Author: andy
 */

#include "T2Transition.h"
#include "MxPolygon.h"
#include "MeshRelationships.h"
#include "MxEdge.h"
#include "MxMesh.h"

static inline PolygonPtr otherPolygon(EdgePtr edge, PolygonPtr poly) {
    int index = edge->polygonIndex(poly);
    int otherIndex = loopIndex(index+1, 2);
    return edge->polygons[otherIndex];
}

HRESULT applyT2PolygonTransition(MeshPtr mesh, PolygonPtr poly)
{
    // go around and check that every edge , check the simplest configuration of
    // each vertex has exactly three edges

    if(!poly || !mesh) {
        return mx_error(E_INVALIDARG, "null arguments");
    }

    if(poly->size() != 3) {
        return mx_error(E_INVALIDARG, "polygon must have exactly three sides");
    }

    // CCW order
    EdgePtr e1 = poly->edges[2];
    EdgePtr e2 = poly->edges[1];
    EdgePtr e3 = poly->edges[0];

    if(e1->polygonCount() != 2 || e2->polygonCount() != 2 || e3->polygonCount() != 2) {
        return mx_error(E_INVALIDARG, "each incident edge of polygon can only be connected to exactly two polygons");
    }

    PolygonPtr p1 = otherPolygon(e1, poly);
    PolygonPtr p2 = otherPolygon(e2, poly);
    PolygonPtr p3 = otherPolygon(e3, poly);
    
    if(p1->size() <= 3 || p2->size() <= 3 || p3->size() <= 3) {
        return mx_error(E_INVALIDARG, "each adjacent polygon must have more than three sides");
    }

    EdgePtr e31; // edge between p3 and p1
    EdgePtr e12; // edge between p1 and p2
    EdgePtr e23; // edge between p2 and p3

    // arbitrarily pick vertex zero as the one to keep
    VertexPtr vert = poly->vertices[0];

    Vector3 centroid = poly->centroid;
    
    std::cout << "poly: " << poly << std::endl;

    // grab the neighboring edges from each polygon and make sure they are
    // the same edge in each adjacent polygon -- make sure this is a manifold
    // polygon
    {
        EdgePtr tmpP1Prev, tmpP1Next;
        VERIFY(getPolygonAdjacentEdges(p1, e1, &tmpP1Prev, &tmpP1Next));

        EdgePtr tmpP2Prev, tmpP2Next;
        VERIFY(getPolygonAdjacentEdges(p2, e2, &tmpP2Prev, &tmpP2Next));

        EdgePtr tmpP3Prev, tmpP3Next;
        VERIFY(getPolygonAdjacentEdges(p3, e3, &tmpP3Prev, &tmpP3Next));
        
        std::cout << "p1: " << p1 << std::endl;
        std::cout << "p2: " << p2 << std::endl;
        std::cout << "p3: " << p3 << std::endl;
        
        std::cout << "e p1 -: " << tmpP1Prev << std::endl;
        std::cout << "e p1 +: " << tmpP1Next << std::endl;
        
        std::cout << "e p2 -: " << tmpP2Prev << std::endl;
        std::cout << "e p2 +: " << tmpP2Next << std::endl;
        
        std::cout << "e p3 -: " << tmpP3Prev << std::endl;
        std::cout << "e p3 +: " << tmpP3Next << std::endl;


        if(tmpP1Prev != tmpP3Next) {
            return mx_error(E_INVALIDARG, "polygons p1 and p3 are not adjacent");
        }

        if(tmpP1Next != tmpP2Prev) {
            return mx_error(E_INVALIDARG, "polygons p1 and p2 are not adjacent");
        }

        if(tmpP3Prev != tmpP2Next) {
            return mx_error(E_INVALIDARG, "polygons p2 and p3 are not adjacent");
        }

        e31 = tmpP1Prev;
        e12 = tmpP2Prev;
        e23 = tmpP3Prev;

        VERIFY(replacePolygonEdgeAndVerticesWithVertex(p1, e1, vert, &tmpP1Prev, &tmpP1Next));
        assert(tmpP1Prev == e31 && tmpP1Next == e12);

        VERIFY(replacePolygonEdgeAndVerticesWithVertex(p2, e2, vert, &tmpP2Prev, &tmpP2Next));
        assert(tmpP2Prev == e12 && tmpP2Next == e23);

        VERIFY(replacePolygonEdgeAndVerticesWithVertex(p3, e3, vert, &tmpP3Prev, &tmpP3Next));
        assert(tmpP3Prev == e23 && tmpP3Next == e31);
    }

    // this call should have no effect, put it here for consistency
    VERIFY(reconnectEdgeVertex(e31, vert, poly->vertices[0]));
    
    VERIFY(reconnectEdgeVertex(e23, vert, poly->vertices[1]));

    VERIFY(reconnectEdgeVertex(e12, vert, poly->vertices[2]));
    
    std::cout << "p1: " << p1 << std::endl;
    std::cout << "p2: " << p2 << std::endl;
    std::cout << "p3: " << p3 << std::endl;

    assert(p1->checkEdges());

    assert(p2->checkEdges());

    assert(p3->checkEdges());

    for(CellPtr cell : poly->cells) {
        VERIFY(disconnectPolygonCell(poly, cell));
        cell->topologyChanged();
    }

    if(poly == mesh->selectedObject()) {
        mesh->selectObject(nullptr, 0);
    }

    mesh->deletePolygon(poly);

    mesh->setPositions(0, 0);

    VERIFY(mesh->positionsChanged());

    return S_OK;
}
