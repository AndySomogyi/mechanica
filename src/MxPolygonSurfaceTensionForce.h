/*
 * MxPolygonSurfaceTensionForce.h
 *
 *  Created on: Sep 20, 2018
 *      Author: andy
 */

#ifndef SRC_MXPOLYGONSURFACETENSIONFORCE_H_
#define SRC_MXPOLYGONSURFACETENSIONFORCE_H_

#include "MxForces.h"

struct MxPolygonSurfaceTensionForce : IForce
{
    virtual HRESULT applyForce(MxObject *obj) const;

    float cellMediaSurfaceTension = 0;
    float cellCellSurfaceTension = 0;
};

#endif /* SRC_MXPOLYGONSURFACETENSIONFORCE_H_ */
