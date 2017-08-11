/*
 * MxModel.h
 *
 *  Created on: Apr 2, 2017
 *      Author: andy
 */

#ifndef SRC_MXMODEL_H_
#define SRC_MXMODEL_H_

#include "mechanica_private.h"


struct MxModel;

/**
 * The type object for a MxSymbol.
 */
MxAPI_DATA(MxType) *MxModel_Type;

struct MxModelMethods {
    void (*execute) (MxModel* cpu);
};

/**
 * Model responsibilities:
 *     * Create and initialize a MxMesh, maintain a reference to the mesh.
 *
 *     * Calculate the total force acting on every element in the mesh.
 *
 *     * Calculate the rate of change of every chemical species in the model,
 *       includes rate of change due to local reactions, and from the
 *       flux between regions contribution.
 */
struct MxModel : MxObject {

    /**
     * The model is responsible for creating a mesh. The mesh is shared
     * between different modules, but owned here.
     */
    struct MxMesh *mesh;


    virtual void foo();





};




void MxModel_init(PyObject *m);

#endif /* SRC_MXMODEL_H_ */
