/*
 * MxSimulator.h
 *
 *  Created on: Feb 1, 2017
 *      Author: andy
 */

#ifndef SRC_MXSIMULATOR_H_
#define SRC_MXSIMULATOR_H_

#include "mechanica_private.h"
#include "MxModel.h"
#include "MxPropagator.h"
#include "MxController.h"
#include "MxView.h"


MxAPI_DATA(MxType*) MxSimulator_Type;

struct MxSimulator : MxObject {
};

HRESULT MxSimulator_init(MxObject *o);

#endif /* SRC_MXSIMULATOR_H_ */
