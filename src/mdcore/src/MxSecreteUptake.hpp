/*
 * MxSecreteUptake.hpp
 *
 *  Created on: Jan 6, 2021
 *      Author: andy
 */

#ifndef SRC_MDCORE_SRC_MXSECRETEUPTAKE_HPP_
#define SRC_MDCORE_SRC_MXSECRETEUPTAKE_HPP_

#include "carbon.h"
#include <set>


HRESULT MxSecrete_AmountToParticles(struct CSpeciesValue* species,
                                    float amount,
                                    uint16_t nr_parts, int32_t *parts,
                                    float *secreted);

HRESULT MxSecrete_AmountWithinDistance(struct CSpeciesValue* species,
                                       float amount,
                                       float radius,
                                       const std::set<short int> *typeIds,
                                       float *secreted);

HRESULT _MxSecreteUptake_Init(PyObject *m);

#endif /* SRC_MDCORE_SRC_MXSECRETEUPTAKE_HPP_ */
