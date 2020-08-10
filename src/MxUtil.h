/*
 * MxParticles.h
 *
 *  Created on: Feb 25, 2017
 *      Author: andy
 */

#ifndef SRC_MXUTIL_H_
#define SRC_MXUTIL_H_

#include "mechanica_private.h"

#include <Magnum/Magnum.h>
#include <Magnum/Math/Color.h>

enum MxPointsType {
    Sphere,
    SolidSphere,
    Disk,
    SolidCube,
    Cube
};

Magnum::Color3 Color3_Parse(const std::string &str);




PyObject* MxPoints(PyObject *m, PyObject *args, PyObject *kwargs);

extern const char* MxColor3Names[];


HRESULT _MxUtil_init(PyObject *m);



#endif /* SRC_MXUTIL_H_ */
