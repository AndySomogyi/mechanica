/*
 * MxPotentialBind.hpp
 *
 *  Created on: Feb 13, 2021
 *      Author: andy
 */

#pragma once
#ifndef SRC_MDCORE_SRC_MXPOTENTIALBIND_HPP_
#define SRC_MDCORE_SRC_MXPOTENTIALBIND_HPP_

#include "MxPotential.h"

/**
 * generalized method to bind a potential to various things
 */
PyObject *MxPotential_Bind(MxPotential *pot, PyObject *args, PyObject *kwargs);

/**
 * bind a potential to a pair of objects.
 *
 * If cluster is true, creates this between a pair of particle types that are
 * in the same cluster.
 */
HRESULT universe_bind_potential(MxPotential *pot, PyObject *a, PyObject *b, bool cluster = false);


#endif /* SRC_MDCORE_SRC_MXPOTENTIALBIND_HPP_ */
