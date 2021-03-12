/*
 * DissapativeParticleDynamics.hpp
 *
 *  Created on: Feb 7, 2021
 *      Author: andy
 */

#pragma once

#include "MxPotential.h"

#ifndef _DISSAPATIVEPARTICLEDYNAMICS_HPP_
#define _DISSAPATIVEPARTICLEDYNAMICS_HPP_


struct DPDPotential : public MxPotential {
    
    // strength of conserative interaction
    float alpha;
    
    // strength of dissapative interaction
    float gamma;
    
    // strength of random interaction
    float sigma;
};

PyObject * DPDPotential_New(float alpha, float gamma, float sigma, float cutoff, bool shifted);


HRESULT _DPDPotential_Init(PyObject *m);



#endif /* _DISSAPATIVEPARTICLEDYNAMICS_HPP_ */
