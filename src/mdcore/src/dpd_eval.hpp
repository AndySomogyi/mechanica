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
#include "potential_eval.hpp"
#include <random>




MX_ALWAYS_INLINE bool dpd_eval(DPDPotential *p, float gaussian,
                               MxParticle *pi, MxParticle *pj, float* dx, float r2 , FPTYPE *energy) {
    
    static const float delta = 1.f / std::sqrt(_Engine.dt);
    
    float cutoff = p->b;
    
    if(r2 > cutoff * cutoff) {
        return false;
    }
    
    float r = std::sqrt(r2);
    
    assert(r >= p->a);
    
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
    
    pj->force = {pj->f[0] - f * e[0], pj->f[1] - f * e[1], pj->f[2] - f * e[2] };
    
    pi->force = {pi->f[0] + f * e[0], pi->f[1] + f * e[1], pi->f[2] + f * e[2] };
    
    // TODO: correct energy
    *energy = 0;
    
    return true;
}

MX_ALWAYS_INLINE bool dpd_boundary_eval(DPDPotential *p, float gaussian,
                               MxParticle *pi, const float *velocity, const float* dx, float r2 , FPTYPE *energy) {
    
    static const float delta = 1.f / std::sqrt(_Engine.dt);
    
    float cutoff = p->b;
    
    if(r2 > cutoff * cutoff) {
        return false;
    }
    
    float r = std::sqrt(r2);
    
    // unit vector
    Magnum::Vector3 e = {dx[0] / r, dx[1] / r, dx[2] / r};
    
    Magnum::Vector3 v = {pi->velocity[0] - velocity[0], pi->velocity[1] - velocity[1], pi->velocity[2] - velocity[2]};
    
    // conservative force
    float omega_c = (1 - r / cutoff);
    
    float fc = p->alpha * omega_c;
    
    // dissapative force
    float omega_d = omega_c * omega_c;
    
    float fd = -p->gamma * omega_d * Magnum::Math::dot(e, v);
    
    float fr = p->sigma * omega_c * delta;
    
    float f = fc + fd + fr;
    
    pi->force = {pi->f[0] + f * e[0], pi->f[1] + f * e[1], pi->f[2] + f * e[2] };
    
    // TODO: correct energy
    *energy = 0;
    
    return true;
}

// MX_ALWAYS_INLINE void potential_eval ( struct MxPotential *p , FPTYPE r2 , FPTYPE *e , FPTYPE *f ) {

MX_ALWAYS_INLINE bool potential_eval_super_ex(const space_cell *cell,
                            MxPotential *pot, MxParticle *part_i, MxParticle *part_j,
                            float *dx, float r2, float number_density, double *epot) {
    
    float e;
    bool result = false;
    
    // if distance is less that potential min distance, define random
    // for repulsive force.
    if(r2 < pot->a * pot->a) {
        dx[0] = space_cell_gaussian(cell->id);
        dx[1] = space_cell_gaussian(cell->id);
        dx[2] = space_cell_gaussian(cell->id);
        float len = std::sqrt(dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]);
        dx[0] = dx[0] * pot->a / len;
        dx[1] = dx[1] * pot->a / len;
        dx[2] = dx[2] * pot->a / len;
        r2 = pot->a * pot->a;
    }
    
    if(pot->kind == POTENTIAL_KIND_DPD) {
        /* update the forces if part in range */
        if (dpd_eval((DPDPotential*)pot, space_cell_gaussian(cell->id), part_i, part_j, dx, r2 , &e)) {
            
            // the number density is a union after the force 3-vector.
            part_i->f[3] += number_density;
            part_j->f[3] += number_density;
            
            /* tabulate the energy */
            *epot += e;
            result = true;
        }
    }
    else {
        float f;
    
        /* update the forces if part in range */
        if (potential_eval_ex(pot, part_i->radius, part_j->radius, r2 , &e , &f )) {
            
            for (int k = 0 ; k < 3 ; k++ ) {
                float w = f * dx[k];
                part_i->f[k] -= w;
                part_j->f[k] += w;
            }
            
            // the number density is a union after the force 3-vector.
            part_i->f[3] += number_density;
            part_j->f[3] += number_density;
            
            /* tabulate the energy */
            *epot += e;
            result = true;
        }
    }

    return result;
}





#endif /* SRC_MDCORE_SRC_DPD_EVAL_HPP_ */
