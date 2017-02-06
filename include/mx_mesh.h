/*
 * mx_mesh.h
 *
 *  Created on: Jan 31, 2017
 *      Author: andy
 */

#ifndef INCLUDE_MX_MESH_H_
#define INCLUDE_MX_MESH_H_

#include "mx_object.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MxMeshData {
  void *pData;

} MxMeshData;

typedef struct MxMesh MxMesh;

/**
 * This instance of MxTypeObject represents the Mechanica module type.
 */
MxAPI_DATA(struct MxType*) MxMesh_Type;

/**
 * Return true if p is a mesh object, or a subtype of a module object.
 */
MxAPI_FUNC(int) MxMesh_Check(MxObject *p);
    
    
    
    
    
#ifdef __cplusplus
}
#endif



#endif /* INCLUDE_MX_MESH_H_ */
