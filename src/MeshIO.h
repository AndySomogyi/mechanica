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


struct IMeshObjectTypeHandler {
    virtual CType *cellType(const char* cellName, int cellIndex) = 0;
    virtual CType *polygonType(int polygonIndex) = 0;
    virtual CType *partialPolygonType(const CType *cellType, const CType *polyType) = 0;
};

MxMesh *MxMesh_FromFile(const char* fname, float density, IMeshObjectTypeHandler *typeHandler);

HRESULT MxMesh_WriteFile(const MxMesh *mesh, const char* fname);

#endif /* SRC_MESHIO_H_ */
