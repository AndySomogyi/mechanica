/*
 * T3Transition.cpp
 *
 *  Created on: Aug 27, 2018
 *      Author: andy
 */

#include <T3Transition.h>

#include <Magnum/Math/Matrix4.h>

HRESULT applyT3PolygonSemiMajorAxisAngleTransition(MeshPtr mesh,
        PolygonPtr poly, float radians, PolygonPtr* p1, PolygonPtr* p2)
{
}

HRESULT applyT3PolygonBisectPlaneTransition(MeshPtr mesh, PolygonPtr poly,
        Vector3* normal, PolygonPtr* p1, PolygonPtr* p2) {
    //Magnum::Math::planeEquation(const Vector3<T>& normal, const Vector3<T>& point)
}
