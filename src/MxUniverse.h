/*
 * MxUniverse.h
 *
 *  Created on: Sep 7, 2019
 *      Author: andy
 */

#ifndef SRC_MXUNIVERSE_H_
#define SRC_MXUNIVERSE_H_

#include "mechanica_private.h"
#include "mdcore_single.h"


struct MxUniverse : PyObject {


    /**
     * MDCore MD engine
     */
    CListWrap potentials;
};


/**
 * The the particle type type
 */
CAPI_DATA(PyTypeObject) MxUniverse_Type;

/**
 * The single global instance of the universe
 */
CAPI_DATA(MxUniverse) _universe;

/**
 * Init and add to python module
 */
void MxUniverse_init(PyObject *m);



#endif /* SRC_MXUNIVERSE_H_ */
