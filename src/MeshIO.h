/*
 * MeshIO.h
 *
 *  Created on: Apr 5, 2018
 *      Author: andy
 */

#ifndef SRC_MESHIO_H_
#define SRC_MESHIO_H_

#include "MxMesh.h"

#include <functional>

typedef std::function<MxCellType* (const char* name, int)> MeshCellTypeHandler;

MxMesh *MxMesh_FromFile(const char* fname, float density, MeshCellTypeHandler cellTypeHandler);

#endif /* SRC_MESHIO_H_ */
