/*
 * T1Transition.h
 *
 *  Created on: Jul 24, 2018
 *      Author: andy
 */

#ifndef SRC_T1TRANSITION_H_
#define SRC_T1TRANSITION_H_

#include "MeshOperations.h"
#include "MxEdge.h"




HRESULT applyT1Edge2Transition(MeshPtr mesh, EdgePtr edge);

HRESULT applyT1Edge3Transition(MeshPtr mesh, EdgePtr edge);

#endif



