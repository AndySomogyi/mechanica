/*
 * XVertex.h
 *
 *  Created on: Dec 10, 2020
 *      Author: andy
 */

#ifndef SRC_MDCORE_SRC_VERTEX_H_
#define SRC_MDCORE_SRC_VERTEX_H_

#include <platform.h>
#include <MxParticle.h>

struct Vertex 
{
};

struct VertexHandle : PyObject
{
    int32_t id;
};

/**
 * vertex is special, it extends particle.
 */
CAPI_DATA(MxParticleType*) Vertex_TypePtr;

HRESULT _vertex_init(PyObject *m);

#endif /* SRC_MDCORE_SRC_XVERTEX_H_ */
