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
#include <Magnum/Magnum.h>
#include <Magnum/Math/Vector3.h>

enum BoundaryConditionKind {
    BOUNDARY_VELOCITY       = 1 << 0,
    BOUNDARY_PERIODIC       = 1 << 1,
    BOUNDARY_FREESLIP       = 1 << 2,
    BOUNDARY_POTENTIAL      = 1 << 3,
    BOUNDARY_NO_SLIP        = 1 << 4, // really just velocity with zero velocity
    BOUNDARY_ACTIVE         = BOUNDARY_FREESLIP | BOUNDARY_VELOCITY | BOUNDARY_POTENTIAL
};

struct MxBoundaryCondition : PyObject {
    BoundaryConditionKind kind;

    // id of this boundary, id's go from 0 to 6 (top, bottom, etc..)
    int id;
    Magnum::Vector3 velocity;

    // restoring percent
    float restore;

    const char* name;

    /**
     * pointer to offset in main array allocated in MxBoundaryConditions.
     */
    struct MxPotential **potenntials;

    // many potentials act on the sum of both particle radii, so this
    // paramter makes it looks like the wall has a sheet of particles of
    // radius.
    float radius;

    /**
     * sets the potential for the given particle type.
     */
    void set_potential(struct MxParticleType *ptype, struct MxPotential *pot);

    std::string str(bool show_name) const;
};

struct MxBoundaryConditions: PyObject {

    MxBoundaryCondition top;
    MxBoundaryCondition bottom;
    MxBoundaryCondition left;
    MxBoundaryCondition right;
    MxBoundaryCondition front;
    MxBoundaryCondition back;

    // pointer to big array of potentials, 6 * max types.
    // each boundary condition has a pointer that's an offset
    // into this array, so allocate and free in single block.
    // allocated in MxBoundaryConditions_Init.
    struct MxPotential **potenntials;

    /**
     * sets a potential for ALL boundary conditions and the given potential.
     */
    void set_potential(struct MxParticleType *ptype, struct MxPotential *pot);

    /**
     * bitmask of periodic boundary conditions
     */
    uint32_t periodic;
};

int MxBoundaryCondition_Check(const PyObject *obj);

int MxBoundaryConditions_Check(const PyObject *obj);

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
