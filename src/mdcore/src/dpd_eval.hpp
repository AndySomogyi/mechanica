/*
 * dpd_eval.hpp
 *
 *  Created on: Feb 7, 2021
 *      Author: andy
 */

#pragma once
#ifndef SRC_MDCORE_SRC_DPD_EVAL_HPP_
#define SRC_MDCORE_SRC_DPD_EVAL_HPP_

#include "MxPotential.h"
#include "DissapativeParticleDynamics.hpp"
#include <random>




MX_ALWAYS_INLINE bool dpd_eval(DPDPotential *p, float gaussian, MxParticle *pi, MxParticle *pj, float* dx, float r2 , FPTYPE *energy) {
    
    static const float delta = 1.f / std::sqrt(_Engine.dt);
    
    float cutoff = p->b;
    
    if(r2 > cutoff * cutoff) {
        return false;
    }
    
    float r = std::sqrt(r2);
    
    // unit vector
    Magnum::Vector3 e = {dx[0] / r, dx[1] / r, dx[2] / r};
    
    Magnum::Vector3 v = pi->velocity - pj->velocity;
    
    // conservative force
    float omega_c = (1 - r / cutoff);
    
    float fc = p->alpha * omega_c;
    
    // dissapative force
    float omega_d = omega_c * omega_c;
    
    float fd = -p->gamma * omega_d * Magnum::Math::dot(e, v);
    
    float fr = p->sigma * omega_c * delta;
    
    float f = fc + fd + fr;
    
    pi->force = {pi->f[0] - f * e[0], pi->f[1] - f * e[1], pi->f[2] - f * e[2] };
    
    pj->force = {pi->f[0] + f * e[0], pi->f[1] + f * e[1], pi->f[2] + f * e[2] };
    
    return true;
}





#endif /* SRC_MDCORE_SRC_DPD_EVAL_HPP_ */
