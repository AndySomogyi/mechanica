/*
 * MxParticles.h
 *
 *  Created on: Feb 25, 2017
 *      Author: andy
 */

#ifndef SRC_MXUTIL_H_
#define SRC_MXUTIL_H_

#include "mechanica_private.h"

enum MxPointsType {
    Sphere,
    SolidSphere,
    Disk
};




PyObject* MxPoints(PyObject *m, PyObject *args, PyObject *kwargs);


HRESULT _MxUtil_init(PyObject *m);



#endif /* SRC_MXUTIL_H_ */
