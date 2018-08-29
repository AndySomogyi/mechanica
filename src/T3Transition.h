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



#endif /* SRC_T3TRANSITION_H_ */
