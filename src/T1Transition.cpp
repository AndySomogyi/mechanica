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
        int v1Index_0 = edge->polygons[0]->vertexIndex(v1);
        int v2Index_0 = edge->polygons[0]->vertexIndex(v2);

#ifndef NDEBUG
        int v1Index_1 = edge->polygons[1]->vertexIndex(v1);
        int v2Index_1 = edge->polygons[1]->vertexIndex(v2);
#endif
        assert(v1Index_0 >= 0 && v2Index_0 >= 0 && v1Index_0 != v2Index_0);

        if(((v1Index_0 + 1) % edge->polygons[0]->size()) == v2Index_0) {
            // found p2
            p2 = edge->polygons[0];
            p4 = edge->polygons[1];
            assert((v2Index_1 + 1) % edge->polygons[1]->size() == v1Index_1);
        }
        else {
            // found p2
            p2 = edge->polygons[0];
            p4 = edge->polygons[1];
            assert((v1Index_1 + 1) % edge->polygons[1]->size() == v2Index_1);
        }
    }

    assert(p4 && p2);

    // iterate over one set of neighboring polygons, just choose p4's neighbors. Then polygon p1 is
    // incident to v1 and adjacent to p2, and polygon p3 is incident to v2 and adjacent to p2.
    for(EdgePtr e : p4->edges) {
        assert(e);
        for(PolygonPtr p : e->polygons) {
            if(p && p != p4 && p != p2) {
                if(!p1 && incidentPolygonVertex(p, v1)) {
                    p1 = p;
                }

                else if(!p3 && incidentPolygonVertex(p, v2)) {
                    p3 = p;
                }

                if(p1 && p3) {
                    break;
                }
            }
        }
    }

    assert(p1 && p3);

    // new vertex positions, divide the line connecting the p2 and p4 centroids
    // into 3 so v1 is 1/3 the way towards p2, and v2 is 2/3 the way towards p2.
    Vector3 centerVec = p2->centroid - p4->centroid;
    float centerDistance = centerVec.length();
    v1->position = p4->centroid + (1/3) * centerVec;
    v2->position = p2->centroid - (1/3) * centerVec;

    VERIFY(disconnectPolygonEdgeVertex(p4, edge, v2));
    VERIFY(disconnectPolygonEdgeVertex(p2, edge, v1));
    VERIFY(insertPolygonEdgeVertex(p1, edge, v2, v1));
    VERIFY(insertPolygonEdgeVertex(p2, edge, v1, v2));
    VERIFY(mesh->positionsChanged());

    return S_OK;
}

HRESULT applyT1Edge3Transition(MeshPtr mesh, EdgePtr edge) {
    return E_NOTIMPL;
}
