/*
 * MxCutPlane.hpp
 *
 *  Created on: Mar 26, 2021
 *      Author: andy
 */
#pragma once
#ifndef SRC_RENDERING_MXCUTPLANE_HPP_
#define SRC_RENDERING_MXCUTPLANE_HPP_

#include <MxParticle.h>
#include <space_cell.h>



/**
 * get a borrowed reference to the cut planes collection.
 */
PyObject *MxClipPlanes_Get();

std::vector<Magnum::Vector4> MxClipPlanes_ParseConfig(PyObject *clipPlanes);

/**
 * internal function to initalize the cut plane types
 */
HRESULT _MxClipPlane_Init(PyObject *m);

#endif /* SRC_RENDERING_MXCUTPLANE_HPP_ */
