/*
 * mx_mesh.h
 *
 *  Created on: Jan 31, 2017
 *      Author: andy
 */

#ifndef INCLUDE_MX_MESH_H_
#define INCLUDE_MX_MESH_H_

#include "carbon.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * A Mechanica mesh currently only supports triangular meshes.
 *
 * Meshes are dynamic, the runtime can deform, or structurally rearange the
 * mesh.
 *
 * Consists of postion, velocity,
 */
typedef struct MxMeshData {
  void *pData;

} MxMeshData;

MxAPI_STRUCT(MxMesh);


/**
 * This instance of CTypeObject represents the Mechanica module type.
 */
CAPI_DATA(struct CType*) MxMesh_Type;

/**
 * Return true if p is a mesh object, or a subtype of a module object.
 */
CAPI_FUNC(int) MxMesh_Check(CObject *p);
    
    
    
    
    
#ifdef __cplusplus
}
#endif



#endif /* INCLUDE_MX_MESH_H_ */
