/*
 * MxUniverse.cpp
 *
 *  Created on: Sep 7, 2019
 *      Author: andy
 */

#include <MxUniverse.h>

MxUniverse universe = {

};


PyTypeObject MxUniverse_Type = {
        PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "Universe",
        .tp_doc = "Custom objects",
        .tp_basicsize = sizeof(MxUniverse),
        .tp_itemsize = 0,
        .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        .tp_new = NULL,
        .tp_members = NULL,
        .tp_descr_get = (descrgetfunc)NULL,
        .tp_init = (initproc)NULL
};


void MxUniverse_init(PyObject* m)
{
}
