/*
 * DifferentialGeometry.h
 *
 *  Created on: Dec 8, 2017
 *      Author: andy
 */

#ifndef SRC_DIFFERENTIALGEOMETRY_H_
#define SRC_DIFFERENTIALGEOMETRY_H_

#include "MxMesh.h"


/**
 * Calculates the mean and gaussian curvatures using the discrete method.
 *
 * The curvature of a vertex is with respect to a surface that belongs to
 * the given cell.
 */
HRESULT discreteCurvature(CCellPtr, CVertexPtr, float *meanCurvature,
        float *gaussianCurvature);



#endif /* SRC_DIFFERENTIALGEOMETRY_H_ */
