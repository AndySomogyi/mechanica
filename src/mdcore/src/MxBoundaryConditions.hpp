/*
 * BoundaryConditions.h
 *
 *  Created on: Feb 10, 2021
 *      Author: andy
 */
#pragma once
#ifndef SRC_MDCORE_SRC_BOUNDARYCONDITIONS_H_
#define SRC_MDCORE_SRC_BOUNDARYCONDITIONS_H_

#include <platform.h>

struct MxBoundaryCondition : PyObject {
    
};

struct MxBoundaryConditions: PyObject {
    
    MxBoundaryCondition top;
    MxBoundaryCondition bottom;
    MxBoundaryCondition left;
    MxBoundaryCondition right;
    MxBoundaryCondition front;
    MxBoundaryCondition back;
    
    /**
     * bitmask of periodic boundary conditions
     */
    uint32_t periodic;
};

/**
 * initialize a boundary condition with either a number that's a bitmask of the
 * BC types, or a dictionary.
 *
 * cells: pointer to 3-vector of cell count, this method will adjust cell count
 * if periodic, make sure cell count is at least 3 in peridic directions. 
 *
 * initializes a boundary conditions bitmask from the stuff in the py dict.
 */

HRESULT MxBoundaryConditions_Init(MxBoundaryConditions *bc, int *cells, PyObject *args);


/**
 * internal method, initialze the types
 */
HRESULT _MxBoundaryConditions_Init(PyObject* m);

#endif /* SRC_MDCORE_SRC_BOUNDARYCONDITIONS_H_ */
