/*
 * MxSymbol.h
 *
 *  Created on: Apr 2, 2017
 *      Author: andy
 */

#ifndef SRC_MXSYMBOL_H_
#define SRC_MXSYMBOL_H_

#include "mechanica_private.h"

struct MxSymbol : CObject {
};

/**
 * Init and add to python module
 */
void MxSymbol_init(PyObject *m);

#endif /* SRC_MXSYMBOL_H_ */
