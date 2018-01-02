/*
 * DifferentialGeometry.h
 *
 *  Created on: Dec 8, 2017
 *      Author: andy
 */

#ifndef SRC_DIFFERENTIALGEOMETRY_H_
#define SRC_DIFFERENTIALGEOMETRY_H_

#include "mechanica_private.h"
#include "MxMesh.h"


/**
 * Calculates the mean and gaussian curvatures using the discrete method.
 *
 * The curvature of a vertex is with respect to a surface that belongs to
 * the given cell.
 */
HRESULT discreteCurvature(CCellPtr, CVertexPtr, float *meanCurvature,
        float *gaussianCurvature);


/**
 * Calculates the mean and gaussian curvatures using the discrete method.
 *
 * The curvature of a vertex is with respect to a surface that belongs to
 * the given cell.
 */
HRESULT discreteCurvature(CCellPtr, CVertexPtr, CTrianglePtr tri,
        float *meanCurvature, float *gaussianCurvature);


float forceDivergence(CVertexPtr v);

/**
 * Calculate a cell's force divergence contribution at a vertex.
 */
float forceDivergenceForCell(CVertexPtr v, CCellPtr c);

/**
 * Constructs a plane from a collection of points
 * so that the summed squared distance to all points is minimzized
 */
HRESULT planeFromPoints(const std::vector<CVertexPtr> &pts, Vector3 &normal, Vector3 &point);


Vector3 centroid(const std::vector<CVertexPtr> &pts);

Vector3 centroidTriangleFan(CVertexPtr center, const std::vector<TrianglePtr> &tri);

#endif /* SRC_DIFFERENTIALGEOMETRY_H_ */
