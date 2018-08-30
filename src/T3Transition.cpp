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
        PolygonPtr poly, float radians, PolygonPtr* p1, PolygonPtr* p2)
{
    return E_NOTIMPL;
}

HRESULT applyT3PolygonBisectPlaneTransition(MeshPtr mesh, PolygonPtr poly,
        Vector3* normal, PolygonPtr* p1, PolygonPtr* p2) {

    // make a plane, point normal form with poly centroid as point
    Vector4 plane = Magnum::Math::planeEquation(*normal, poly->centroid);

    // index of the two split edges
    int e1 = -1, e2 = -1;

    for(int i = 0; i < poly->edges.size(); ++i) {
        CEdgePtr e = poly->edges[i];
        float d0 = 	Math::Distance::pointPlaneScaled(e->vertices[0]->position, plane);
        float d1 = 	Math::Distance::pointPlaneScaled(e->vertices[1]->position, plane);

        if(d0 == 0. || d1 == 0.) {
            return mx_error(E_FAIL, "polygon vertex intersect exactly with cut plane");
        }

        if(d0 * d1 < 0.) {
            if(e1 < 0) {
                e1 = i;
            }
            else if(e2 < 0) {
                e2 = i;
                break;
            }
        }
    }

    if(e1 < 0 || e2 < 0) {
        return mx_error(E_FAIL, "cut plane does not intersect at least two polygon edges");
    }









    return E_NOTIMPL;
}
