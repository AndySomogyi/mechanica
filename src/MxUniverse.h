/*
 * MxUniverse.h
 *
 *  Created on: Sep 7, 2019
 *      Author: andy
 */

#ifndef SRC_MXUNIVERSE_H_
#define SRC_MXUNIVERSE_H_

#include "mechanica_private.h"
#include "engine.h"


struct MxUniverse : CObject {


    /**
     * MDCore MD engine
     */
    engine engine;
};

/**
 * Init and add to python module
 */
void MxUniverse_init(PyObject *m);



#endif /* SRC_MXUNIVERSE_H_ */
