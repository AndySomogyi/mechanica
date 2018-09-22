/*
 * MxPartialPolygonSurfaceTensionForce.cpp
 *
 *  Created on: Sep 20, 2018
 *      Author: andy
 */

#include <MxPolygonSurfaceTensionForce.h>
#include "MxCell.h"

HRESULT MxPolygonSurfaceTensionForce::applyForce(MxObject* obj) const
{

    float k = 0;
    MxPolygon *pp = static_cast<MxPolygon*>(obj);

    if(pp->cells[0]->isRoot() || pp->cells[1]->isRoot()) {
        k = cellMediaSurfaceTension;
    } else {
        k = cellCellSurfaceTension;
    }

    for(uint i = 0; i < pp->vertices.size(); ++i) {
        VertexPtr vi = pp->vertices[i];
        VertexPtr vn = pp->vertices[(i+1)%pp->vertices.size()];
        Vector3 dx = vn->position - vi->position;

        vi->force += k * dx;
        vn->force -= k * dx;
    }

    return S_OK;
}
