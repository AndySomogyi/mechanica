/*
 * MeshOperations.h
 *
 *  Created on: Nov 20, 2017
 *      Author: andy
 */

#ifndef SRC_MESHOPERATIONS_H_
#define SRC_MESHOPERATIONS_H_

#include "MxMeshCore.h"
#include "MxMesh.h"
#include "MxPolygon.h"





/**
 * Basic H to I and I to H transitions according to Honda.
 *
 * (a) we know all edge lengths as edges are explicit objects.
 *
 * In general, an edge may be connected to either two or three polygons. The H to I
 * and I to H transitions are only valid for volumetric regions, where a edge is
 * incident to three polygons.
 *
 * If an edge is incident to three polygons, we continue checking if an H to I or I to H
 * transition is valid.
 *
 * If an edge is incident to three polygons, and none of the polygons is a triangle
 * (i.e. all of three incident polygons have four or more edges), then the edge is
 * a type I configuration. A type I configuration can be 'flipped' to an H configuration
 * where the edge is replaced with a triangle.
 *
 * If one of the three polygons is a triangle, then that triangle is a type H configuration. This
 * configuration can be 'flipped' to an I configuration by replacing the triangle with a edge.
 *
 * If a triangle connects two cells, and those two cells are only connected by that
 * triangle, i.e. if the all of the edges of the triangle are incident to these
 * two cells and the root cell, then the triangle can not be replaced with an edge.
 *
 * The H to I transition is intended to flip a triangle to an edge, as such all of the
 * incident edges to the triangle will be removed. Consider the case where the
 * Candidate triangle is adjacent to another triangle. The only way to flip the
 * candidate triangle to an edge is to also replace the adjacent triangle
 * with an edge. The two H to I and I to H transition are intended to be purely
 * local do not have non-local side effects. Therefore, we require that a the
 * triangle in a valid H to I transition must be adjacent to three polygons
 * with at least four sides each. Each of these three polygons will have
 * a side (edge) removed.
 *
 *
 *
 *
 */
HRESULT Mx_FlipPolygonToEdge(MeshPtr mesh, PolygonPtr poly, EdgePtr *edge);

HRESULT Mx_FlipEdgeToPolygon(MeshPtr mesh, EdgePtr edge, PolygonPtr *poly);

bool Mx_IsPolygonToEdgeConfiguration(CEdgePtr edge);

bool Mx_IsEdgeToPolygonConfiguration(CEdgePtr edge);

HRESULT Mx_FlipEdge(MeshPtr mesh, EdgePtr edge);


HRESULT Mx_CollapsePolygon(MeshPtr mesh, PolygonPtr poly);


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
HRESULT Mx_SplitPolygonSemiMajorAxisAngle(MeshPtr mesh, PolygonPtr poly,
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
HRESULT Mx_SplitPolygonBisectPlane(MeshPtr mesh, PolygonPtr poly,
        Vector3 *normal, PolygonPtr *p1, PolygonPtr *p2);




/**
 * Splits a edge into two, and updates all of the incident polygons with the new edge.
 * Creates a new edge, ep, and inserts that into all of the incident polygons.
 *
 * @param mesh
 * @param e
 * @param v: an optional vertex. If this argument is provided, this vertex will be
 *           inserted into the middle of this edge. If this vertex is not provided,
 *           Mx_SplitEdge will create a new vertex at the midpoint of edge e, and
 *           insert that instead.
 * @param ep: an out parameter, this is the newly created edge that is inserted into
 *           the surrounding polygons.
 * @param vp: the vertex that was placed in the middle of this edge. If v was provided,
 *           then this gets simply set to v, otherwise this is the newly created vertex.
 */
HRESULT Mx_SplitEdge(MeshPtr mesh, EdgePtr e, VertexPtr v, EdgePtr *ep, VertexPtr *vp);


HRESULT Mx_SplitCell(MeshPtr mesh, CellPtr c,
        float* planeEqn, CellPtr* c1, CellPtr* c2);

#endif /* SRC_MESHOPERATIONS_H_ */
