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
    EdgePtr e1a = nullptr, e2a = nullptr;

    // the pair of edges that are after (CCW winding relative to poly) thar
    // are formed when e1a and e2a get split with a new vertex.
    EdgePtr e1b = nullptr, e2b = nullptr;


    for(int i = 0; i < poly->edges.size(); ++i) {
        EdgePtr e = poly->edges[i];
        float d0 = 	Math::Distance::pointPlaneScaled(e->vertices[0]->position, plane);
        float d1 = 	Math::Distance::pointPlaneScaled(e->vertices[1]->position, plane);

        if(d0 == 0. || d1 == 0.) {
            return mx_error(E_FAIL, "polygon vertex intersect exactly with cut plane");
        }

        if(d0 * d1 < 0.) {
            if(!e1a) {
                e1a = e;
            }
            else if(!e2a) {
                e2a = e;
                break;
            }
        }
    }

    if(!e1a || !e2a) {
        return mx_error(E_FAIL, "cut plane does not intersect at least two polygon edges");
    }

    // create two new vertices that are at the center of the two edges to split.
    VertexPtr v1 = mesh->createVertex((e1a->vertices[0]->position + e1a->vertices[1]->position) / 2., MxVertex_Type);
    VertexPtr v2 = mesh->createVertex((e2a->vertices[0]->position + e2a->vertices[1]->position) / 2., MxVertex_Type);

    // grab the two adjacent polygons that contact this poly via the two edges to be
    // bisected.
    PolygonPtr p1 = otherPolygon2D(e1a, poly);
    PolygonPtr p2 = otherPolygon2D(e2a, poly);

    if(!p1 || !p2) {
        return mx_error(E_FAIL, "could not locate adjacent polygons");
    }


    return E_NOTIMPL;
}
