/*
 * Cell.h
 *
 *  Created on: Dec 10, 2020
 *      Author: andy
 */

#ifndef SRC_MDCORE_SRC_CELL_H_
#define SRC_MDCORE_SRC_CELL_H_

#include <platform.h>

struct Cell
{
};

struct CellHandle : PyObject
{
};

/**
 */
CAPI_DATA(PyTypeObject) Cell_Type;

HRESULT _cell_init(PyObject *m);

#endif /* SRC_MDCORE_SRC_CELL_H_ */
