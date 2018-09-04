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


/**
 * splits a polygon along a cut plane relative to the polygon's semi-major axis. We
 * determine the principal axis of the polygon by first calculating the intertia tensor,
 * then taking the eigen vectors of it. The eigen vectors define the semi-major, semi-minor
 * and perpendicular axes -- the eigen vectors of the intertia tensor defind a local
 * coordinate of the polygon.
 *
 * Note, all vertex dynamics are computed in world space, the local coordinate
 * system is a derived quantity of a polygon, a polygon does not maintain, track or
 * persist a local coordinate axes in any way.
 *
 * Returns two new polygons in p1 and p2, note, it is possible that the original cell is
 * returned in either p1 or p2, or future versions might destroy the poly polygon and
 * generate two new polygons.
 */
HRESULT applyT3PolygonSemiMajorAxisAngleTransition(MeshPtr mesh, PolygonPtr poly,
        float radians, PolygonPtr *p1, PolygonPtr *p2);

/**
 * Apply a T3 (split) operation to a polygon where the vertices are bisected
 * by a plane (in world space). The plane is specified in point-normal form,
 * where the point is the center of mass of this polygon, and the normal is
 * given in normal.
 *
 * The plane normal should be roughly perpendicular to the face normal
 * of the polygon. If the plane does not bisect any vertices, then an error
 * is returned. The vertices of the polygon are partitioned into two sections:
 * above and below the plane, and the polygon is split between between these
 * two sections.
 *
 * returns the two (potentially) new polygons in p1 and p2, returns S_OK on
 * Success, or E_INVALIDARG in an error conditions. Error conditions may be
 * when the plane does not bisect at least two edges. Note, if a vertex is
 * Exactly on the split plane (highly unlikely), it is perturbed in a
 * random direction perpendicular to the plane.
 *
 * Currently only supports convex polygons *well*, if a polygon is concave, and
 * there are more than two edges that cross the bisection plane, this procedure
 * chooses the first two edges that it encounters.
 */
HRESULT applyT3PolygonBisectPlaneTransition(MeshPtr mesh, PolygonPtr poly,
        Vector3 *normal, PolygonPtr *p1, PolygonPtr *p2);



#endif /* SRC_T3TRANSITION_H_ */
