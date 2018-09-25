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


typedef HRESULT (*MxObjectChangedHandler)(MxObject *pThis, const MxObject *obj, uint32_t what);

struct MxObjectChangedHolder {
    void *userData;
    MxObjectChangedHandler callBack;
};


/**
 * Define the Mechanica base MxObject to have the same binary layout as
 * python objects.
 *
 * Nothing is actually declared to be a MxObject, but every pointer to
 * a Mechanica object can be cast to a MxObject* (and hence a PyObject*).
 *
 * The python PyObject is laid out as :
 *
 *  struct PyObject {
 *      int32_t ob_refcnt;
 *     _typeobject *ob_type;
 *  }
 *
 */
struct MxObject
{
    Py_ssize_t ob_refcnt;
    struct MxType *ob_type;

    MxObject(struct MxType *type) {
        this->ob_type = type;
        this->ob_refcnt = 1;
    }

    MxObject() {
        this->ob_type = nullptr;
        this->ob_refcnt = 1;
    }

    static MxType *type() { return MxObject_Type; };
};

HRESULT MxObject_ChangeType(MxObject *obj, const MxType *type);


/**
 * Init and add to python module
 */
void MxObject_init(PyObject *m);



#endif /* SRC_MXOBJECT_H_ */
