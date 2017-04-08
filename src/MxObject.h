/*
 * MxObject.h
 *
 *  Created on: Apr 2, 2017
 *      Author: andy
 */

#ifndef SRC_MXOBJECT_H_
#define SRC_MXOBJECT_H_

#include <Python.h>

// public MxObject api
#include "mx_object.h"

/**
 * Define the Mechanica base MxObject to have the same binary layout as
 * python objects.
 *
 * Nothing is actually declared to be a MxObject, but every pointer to
 * a Mechanica object can be cast to a MxObject* (and hence a PyObject*).
 */
struct MxObject : _object {};





/**
 * Init and add to python module
 */
void MxObject_init(PyObject *m);



#endif /* SRC_MXOBJECT_H_ */
