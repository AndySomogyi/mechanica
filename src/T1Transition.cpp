/*
 * T1Transition.cpp
 *
 *  Created on: Jul 24, 2018
 *      Author: andy
 */

#include "T1Transition.h"
#include "MxPolygon.h"

HRESULT applyT1Edge2Transition(MeshPtr mesh, EdgePtr edge) {

    if(edge->polygonCount() != 2) {
        return mx_error(E_FAIL, "edge polygon count must be 2");
    }

    if(edge->polygons[0]->edgeCount() <= 3 || edge->polygons[1]->edgeCount() <= 3) {
        return mx_error(E_FAIL, "can't collapse edge that's connected to polygons with less than 3 sides");

    }

    return E_NOTIMPL;
}

HRESULT applyT1Edge3Transition(MeshPtr mesh, EdgePtr edge) {
    return E_NOTIMPL;
}
