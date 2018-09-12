/*
 * T3Transition.cpp
 *
 *  Created on: Aug 27, 2018
 *      Author: andy
 */

#include <T3Transition.h>
#include <Magnum/Math/Matrix4.h>
#include <Magnum/Math/Distance.h>

HRESULT applyT3PolygonSemiMajorAxisAngleTransition(MeshPtr mesh,
        PolygonPtr poly, float radians, PolygonPtr* pn1, PolygonPtr* pn2)
{
    return E_NOTIMPL;
}

/**
 * find the other polygon attached to a 2D edge that's not this polygon
 */
static PolygonPtr otherPolygon2D(EdgePtr e, PolygonPtr p) {
    if(e->polygonCount() != 2) {
        return nullptr;
    }
    return e->polygons[0] == p ? e->polygons[1] : e->polygons[0];
}

HRESULT applyT3PolygonBisectPlaneTransition(MeshPtr mesh, PolygonPtr poly,
        Vector3* normal, PolygonPtr* pn1, PolygonPtr* pn2) {

    // make a plane, point normal form with poly centroid as point
    Vector4 plane = Magnum::Math::planeEquation(*normal, poly->centroid);

    // the pair of edges on the base polygon that are bisected by the plane.
    EdgePtr en = nullptr, em = nullptr;

    // the pair of edges that are after (CCW winding relative to poly) thar
    // are formed when e1a and e2a get split with a new vertex.
    EdgePtr en1 = nullptr, em1 = nullptr;

    // the new center splitting edge
    EdgePtr es = nullptr;

    // the two vertices v_{n+1}, v_{m+1} on the tail end of the edge (followed around CCW order on the
    // center polygon).
    // also the two new end vertices, v_{n+2}, v_{m_2} that will have the same positions as the two existing
    // end vertices.
    VertexPtr vn = nullptr, vn1 = nullptr, vn2 = nullptr,
              vm = nullptr, vm1 = nullptr, vm2 = nullptr;

    // indices of the found edges
    int in = -1, im = -1, in1 = -1, im1 = -1;


    for(int i = 0; i < poly->edges.size(); ++i) {
        EdgePtr e = poly->edges[i];
        float d0 = 	Math::Distance::pointPlaneScaled(e->vertices[0]->position, plane);
        float d1 = 	Math::Distance::pointPlaneScaled(e->vertices[1]->position, plane);

        if(d0 == 0. || d1 == 0.) {
            return mx_error(E_FAIL, "polygon vertex intersect exactly with cut plane");
        }

        if(d0 * d1 < 0.) {
            if(!en) {
                in = i;
                in1 = loopIndex(i+1, poly->vertices.size());
                en = e;
                vn =  poly->vertices[i];
                vn2 = wrappedAt(poly->vertices, i+1);
                assert(connectedEdgeVertex(en, vn));
                assert(connectedEdgeVertex(en, vn2));
            }
            else if(!em) {
                im = i;
                im1 = loopIndex(i+1, poly->vertices.size());
                em = e;
                vm =  poly->vertices[i];
                vm2 = wrappedAt(poly->vertices, i+1);
                assert(connectedEdgeVertex(em, vm));
                assert(connectedEdgeVertex(em, vm2));
                break;
            }
        }
    }

    if(!en || !em) {
        return mx_error(E_FAIL, "cut plane does not intersect at least two polygon edges");
    }

    std::cout << "found edges to split: " << std::endl;
    std::cout << "en: " << en << std::endl;
    std::cout << "em: " << em << std::endl;

    // grab the two adjacent polygons that contact this poly via the two edges to be
    // bisected.
    PolygonPtr p1 = otherPolygon2D(en, poly);
    PolygonPtr p2 = otherPolygon2D(em, poly);

    if(!p1 || !p2) {
        return mx_error(E_FAIL, "could not locate adjacent polygons");
    }

    std::cout << "poly p: " << poly << std::endl;
    std::cout << "poly p1: " << p1 << std::endl;
    std::cout << "poly p2: " << p2 << std::endl;


    // create two new vertices that are at the same position as the end vertices,
    // we shrink the edges so the end vertices are at the previous midpoint.
    vn1 = mesh->createVertex((vn->position + vn2->position) / 2., MxVertex_Type);
    vm1 = mesh->createVertex((vm->position + vm2->position) / 2., MxVertex_Type);

    // make the new edges
    en1 = mesh->createEdge(MxEdge_Type, vn1, vn2);
    em1 = mesh->createEdge(MxEdge_Type, vm1, vm2);
    es = mesh->createEdge(MxEdge_Type, vn1, vm1);

    std::cout << "edge en: " << en << std::endl;
    std::cout << "edge em: " << em << std::endl;
    std::cout << "edge en1: " << en1 << std::endl;
    std::cout << "edge em1: " << em1 << std::endl;
    std::cout << "edge es: " << es << std::endl;

    assert(en1 && em1);

    // insert the edges into the two neighboring polygons
    VERIFY(splitPolygonEdge(p1, en1, en));
    VERIFY(splitPolygonEdge(p2, em1, em));

    std::cout << "split p1 and p2 edges: " << std::endl;
    std::cout << "poly p1: " << p1 << std::endl;
    std::cout << "poly p2: " << p2 << std::endl;

    // insert the edges into polygon to be split
    VERIFY(splitPolygonEdge(poly, en1, en));
    VERIFY(splitPolygonEdge(poly, em1, em));

    std::cout << "split poly edges: " << std::endl;
    std::cout << "poly poly: " << poly << std::endl;

    // reconnect the vertex pointers on the old edges
    VERIFY(reconnectEdgeVertex(en, vn1, vn2));
    VERIFY(reconnectEdgeVertex(em, vm1, vm2));

    std::cout << "after reconnecting edge vertices: " << std::endl;
    std::cout << "poly p1: " << p1 << std::endl;
    std::cout << "poly p2: " << p2 << std::endl;
    std::cout << "poly poly: " << poly << std::endl;
    std::cout << "ready to split with new edge: " << es << std::endl;

    // the new daughter polygon, initially empty
    PolygonPtr pn = mesh->createPolygon((MxPolygonType*)poly->ob_type);

    {
        // tmp data structures to copy the elements into,
        // cheaper to copy than to delete and move,
        // simpler code wise also.
        std::vector<VertexPtr> tmpVertices;
        std::vector<MxEdge*> tmpEdges;

        const int vn1Index = indexOf(poly->vertices, vn1);
        const int vm1Index = indexOf(poly->vertices, vm1);
        const int vn2Index = indexOf(poly->vertices, vn2);
        const int vm2Index = indexOf(poly->vertices, vm2);


        for(int i = vm1Index; i != vn2Index; i = loopIndex(i+1, poly->size())) {
            tmpVertices.push_back(poly->vertices[i]);
            if(loopIndex(i+1, poly->size()) != vn2Index) {
                tmpEdges.push_back(poly->edges[i]);
            }
        }
        tmpEdges.push_back(es);
        VERIFY(es->insertPolygon(poly));

        for(int i = vn1Index; i != vm2Index; i = loopIndex(i+1, poly->size())) {
            pn->vertices.push_back(poly->vertices[i]);
            if(loopIndex(i+1, poly->size()) != vm2Index) {
                pn->edges.push_back(poly->edges[i]);
            }
        }
        pn->edges.push_back(es);
        VERIFY(es->insertPolygon(pn));

        poly->vertices = std::move(tmpVertices);
        poly->edges = std::move(tmpEdges);
    }

    poly->_vertexAreas.resize(poly->vertices.size());
    poly->_vertexNormals.resize(poly->vertices.size());

    pn->_vertexAreas.resize(pn->vertices.size());
    pn->_vertexNormals.resize(pn->vertices.size());

    pn->cells = poly->cells;

    assert(poly->checkEdges());
    assert(p1->checkEdges());
    assert(p2->checkEdges());
    assert(pn->checkEdges());

    for(int i = 0; i < 2; ++i) {
        pn->cells[i]->surface.push_back(&pn->partialPolygons[i]);
        pn->cells[i]->topologyChanged();
    }

    VERIFY(mesh->positionsChanged());

    std::cout << "resized polygons:" << std::endl;
    std::cout << "poly p: " << poly << std::endl;
    std::cout << "poly pn: " << pn << std::endl;
    std::cout << "poly p1: " << p1 << std::endl;
    std::cout << "poly p2: " << p2 << std::endl;


    return S_OK;
}
