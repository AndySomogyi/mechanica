/*
 * MxExpression.h
 *
 *  Created on: Apr 4, 2018
 *      Author: andy
 */

#ifndef SRC_MXEXPRESSION_H_
#define SRC_MXEXPRESSION_H_

#include "mechanica_private.h"

struct MxExpression : CObject
{
    CObject *head;
    int32_t length;
    CObject *items[0];
};



#endif /* SRC_MXEXPRESSION_H_ */
