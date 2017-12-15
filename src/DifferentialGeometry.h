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



#endif /* SRC_DIFFERENTIALGEOMETRY_H_ */
