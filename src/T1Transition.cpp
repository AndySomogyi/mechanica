/*
 * T1Transition.cpp
 *
 *  Created on: Jul 24, 2018
 *      Author: andy
 */

#include "T1Transition.h"
#include "MxPolygon.h"
#include "MeshRelationships.h"
#include "MxMesh.h"

HRESULT applyT1Edge2Transition(MeshPtr mesh, EdgePtr edge) {
    
    std::cout << "applyT1Edge2Transition(edge=" << edge << ")" << std::endl;

    if(edge->polygonCount() != 2) {
        return mx_error(E_FAIL, "edge polygon count must be 2");
    }

    if(edge->polygons[0]->size() <= 3 || edge->polygons[1]->size() <= 3) {
        return mx_error(E_FAIL, "can't collapse edge that's connected to polygons with less than 3 sides");

    }

    PolygonPtr p1 = nullptr, p2 = nullptr, p3 = nullptr, p4 = nullptr;

    // top and bottom vertices
    VertexPtr v1 = edge->vertices[0];
    VertexPtr v2 = edge->vertices[1];

    // identify the two side polygons as p4 on the left, and p2 on the right, check
    // vertex winding. Choose p2 to have v2 after v1, and p4 to have v1 after v2
    {
        // TODO use next vert instead of looking at all vertices.
        int v1Index_0 = edge->polygons[0]->vertexIndex(v1);
        int v2Index_0 = edge->polygons[0]->vertexIndex(v2);

#ifndef NDEBUG
        int v1Index_1 = edge->polygons[1]->vertexIndex(v1);
        int v2Index_1 = edge->polygons[1]->vertexIndex(v2);
#endif
        assert(v1Index_0 >= 0 && v2Index_0 >= 0 && v1Index_0 != v2Index_0);

        if(((v1Index_0 + 1) % edge->polygons[0]->size()) == v2Index_0) {
            // found p2 CCW
            p2 = edge->polygons[0];
            p4 = edge->polygons[1];
            assert((v2Index_1 + 1) % edge->polygons[1]->size() == v1Index_1);
        }
        else {
            // found p4
            p4 = edge->polygons[0];
            p2 = edge->polygons[1];
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

    assert(edge->polygonCount() == 0);
    
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

HRESULT applyT1Edge3Transition(MeshPtr mesh, EdgePtr edge) {
    return E_NOTIMPL;
}
