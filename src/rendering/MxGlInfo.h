/*
 * MxGlInfo.h
 *
 *  Created on: Apr 22, 2020
 *      Author: andy
 */

#ifndef SRC_RENDERING_MXGLINFO_H_
#define SRC_RENDERING_MXGLINFO_H_

#include "Mechanica.h"

PyObject *Mx_GlInfo(PyObject *args, PyObject *kwds);

PyObject *Mx_EglInfo(PyObject *args, PyObject *kwds);

std::string gl_info();

#endif /* SRC_RENDERING_MXGLINFO_H_ */
