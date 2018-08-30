/*
 * T3Transition.h
 *
 *  Created on: Aug 27, 2018
 *      Author: andy
 */

#ifndef SRC_T3TRANSITION_H_
#define SRC_T3TRANSITION_H_

#include "MxMesh.h"
#include "MxPolygon.h"

HRESULT applyT3PolygonSemiMajorAxisAngleTransition(MeshPtr mesh, PolygonPtr poly,
        float radians, PolygonPtr *p1, PolygonPtr *p2);

/**
 * Apply a T3 (split) operation to a polygon where the vertices are bisected
 * by a plane. The plane is specified in point-normal form, where the point is
 * the center of mass of this polygon, and the normal is given in normal.
 *
 * The plane normal should be roughly perpendicular to the face normal
 * of the polygon. If the plane does not bisect any vertices, then an error
 * is returned. The vertices of the polygon are partitioned into two sections:
 * above and below the plane, and the polygon is split between between these
 * two sections.
 */
HRESULT applyT3PolygonBisectPlaneTransition(MeshPtr mesh, PolygonPtr poly,
        Vector3 *normal, PolygonPtr *p1, PolygonPtr *p2);



#endif /* SRC_T3TRANSITION_H_ */
