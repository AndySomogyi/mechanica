/*
 * MxModule.h
 *
 *  Created on: Feb 1, 2017
 *      Author: andy
 */

#ifndef SRC_MXMODULE_H_
#define SRC_MXMODULE_H_

#include "mechanica_private.h"

struct MxModule : MxObject {
};

void MxModule_init(PyObject *m);

#endif /* SRC_MXMODULE_H_ */
