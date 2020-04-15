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
#include "MxApplication.h"

enum MxSimulator_Key {
    MXSIMULATOR_NONE,
    MXSIMULATOR_WINDOWLESS,
    MXSIMULATOR_GLFW
};

struct MxSimulator_ConfigurationItem {
    uint32_t key;
    union {
        int intVal;
        int intVecVal[4];
    };
};



CAPI_DATA(PyTypeObject) MxSimulator_Type;

struct MxSimulator : _object {
    int32_t kind;
    void *applicaiton;
};


/**
 * The global simulator object
 */
CAPI_DATA(MxSimulator*) Mx_Simulator;

/**
 * Creates a new simulator if the global one does not exist,
 * returns the global if it does.
 *
 * items: an array of config items, at least one.
 */
CAPI_FUNC(MxSimulator*) MxSimulator_New(MxSimulator_ConfigurationItem *items);

CAPI_FUNC(MxSimulator*) MxSimulator_Get();



// internal method to initialize the simulator type.
HRESULT MxSimulator_init(PyObject *o);

#endif /* SRC_MXSIMULATOR_H_ */
