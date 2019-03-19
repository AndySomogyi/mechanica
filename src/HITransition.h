/*
 * HITransition.h
 *
 *  Created on: Jan 16, 2019
 *      Author: andy
 */

#ifndef SRC_HITRANSITION_H_
#define SRC_HITRANSITION_H_

#include "MeshOperations.h"
#include "MxEdge.h"

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
HRESULT applyHtoITransition(MeshPtr mesh, PolygonPtr poly, EdgePtr *edge);

HRESULT applyItoHTransition(MeshPtr mesh, EdgePtr edge, PolygonPtr *poly);

bool isHConfiguration(CEdgePtr edge);

bool isIConfiguration(CEdgePtr edge);


#endif /* SRC_HITRANSITION_H_ */
