/*
 * MxRenderTarget.h
 *
 *  Created on: Oct 1, 2018
 *      Author: andy
 */

#ifndef SRC_MXRENDERTARGET_H_
#define SRC_MXRENDERTARGET_H_

#include "mechanica_private.h"
#include "MxModel.h"
#include "MxPropagator.h"
#include "MxController.h"
#include "MxView.h"

struct MxRenderTarget: public MxObject
{
public:
    MxRenderTarget();
    virtual ~MxRenderTarget();
};

#endif /* SRC_MXRENDERTARGET_H_ */
