/*
 * Edge.h
 *
 *  Created on: Dec 10, 2020
 *      Author: andy
 */

#ifndef SRC_MDCORE_SRC_EDGE_H_
#define SRC_MDCORE_SRC_EDGE_H_


#include <Vertex.hpp>

struct Edge
{
};

struct EdgeHandle : PyObject
{
    int32_t id;
};



/**
 */
CAPI_DATA(PyTypeObject) Edge_Type;

HRESULT _edge_init(PyObject *m);



#endif /* SRC_MDCORE_SRC_EDGE_H_ */
