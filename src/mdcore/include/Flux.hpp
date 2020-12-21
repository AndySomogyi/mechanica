/*
 * Flux.h
 *
 *  Created on: Dec 21, 2020
 *      Author: andy
 */

#ifndef SRC_MDCORE_SRC_FLUX_H_
#define SRC_MDCORE_SRC_FLUX_H_

#include "platform.h"
#include "mdcore_single_config.h"



struct MxFlux : PyVarObject
{
    int32_t indices[1];
};


CAPI_DATA(PyTypeObject) MxFlux_Type;

#endif /* SRC_MDCORE_SRC_FLUX_H_ */
