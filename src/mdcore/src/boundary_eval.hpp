/*
 * boundary_eval.hpp
 *
 *  Created on: Feb 11, 2021
 *      Author: andy
 */
#pragma once
#ifndef SRC_MDCORE_SRC_BOUNDARY_EVAL_HPP_
#define SRC_MDCORE_SRC_BOUNDARY_EVAL_HPP_

#include "MxParticle.h"
#include "MxPotential.h"
#include "space_cell.h"
#include "engine.h"
#include "dpd_eval.hpp"
#include "potential_eval.hpp"

#include <iostream>

// velocity boundary conditions:
//
// r_new = r_old + 2 d n_w,
// where d is distance particle penetrated into wall, and
// n_w is normal vector into simulation domain.
//
// v_new = 2 U_wall - v_old
// where U_wall is wall velocity.


MX_ALWAYS_INLINE bool boundary_update_pos_vel(MxParticle *p, space_cell *c) {
    
    #define ENFORCE_FREESLIP_LOW(i)                              \
        p->position[i] = -p->position[i] * restitution;          \
        p->velocity[i] *= -restitution;                          \
        enforced = true;                                         \

    #define ENFORCE_FREESLIP_HIGH(i)                             \
        p->position[i] = c->dim[i] - (p->position[i] - c->dim[i]) * restitution;          \
        p->velocity[i] *= -restitution;                          \
        enforced = true;                                         \

   #define ENFORCE_VELOCITY_LOW(i, bc)  \
        p->position[i] = -p->position[i] * bc.restore;  \
        p->velocity = 2.f * bc.velocity - (p->velocity * bc.restore); \
        enforced = true; \

    #define ENFORCE_VELOCITY_HIGH(i, bc)  \
        p->position[i] = 2.f * c->dim[i] - (p->position[i] * bc.restore);        \
        p->velocity = 2.f * bc.velocity - (p->velocity * bc.restore); \
        enforced = true;  \

    
    static const MxBoundaryConditions *bc = &_Engine.boundary_conditions;
    
    static const float restitution = 1.0;
    
    /* Enforce particle position to be within the given boundary */
    bool enforced = false;
    
    if(c->flags & cell_boundary_left && p->x[0] <= 0) {
        switch (bc->left.kind) {
            case BOUNDARY_FREESLIP:
                ENFORCE_FREESLIP_LOW(0);
                break;
            case BOUNDARY_VELOCITY:
                ENFORCE_VELOCITY_LOW(0, bc->left);
                break;
            default:
                break;
        }
    }
    
    if(c->flags & cell_boundary_right && p->x[0] >= c->dim[0]) {
        switch (bc->right.kind) {
            case BOUNDARY_FREESLIP:
                ENFORCE_FREESLIP_HIGH(0);
                break;
            case BOUNDARY_VELOCITY:
                ENFORCE_VELOCITY_HIGH(0, bc->right);
                break;
            default:
                break;
        }
    }
    
    if(c->flags & cell_boundary_front && p->x[1] <= 0) {
        switch (bc->front.kind) {
            case BOUNDARY_FREESLIP:
                ENFORCE_FREESLIP_LOW(1);
                break;
            case BOUNDARY_VELOCITY:
                ENFORCE_VELOCITY_LOW(1, bc->front);
                break;
            default:
                break;
        }
    }
    
    if(c->flags & cell_boundary_back && p->x[1] >= c->dim[1]) {
        switch (bc->back.kind) {
            case BOUNDARY_FREESLIP:
                ENFORCE_FREESLIP_HIGH(1);
                break;
            case BOUNDARY_VELOCITY:
                ENFORCE_VELOCITY_HIGH(1, bc->back);
                break;
            default:
                break;
        }
    }
    
    if(c->flags & cell_boundary_bottom && p->x[2] <= 0) {
        switch (bc->bottom.kind) {
            case BOUNDARY_FREESLIP:
                ENFORCE_FREESLIP_LOW(2);
                break;
            case BOUNDARY_VELOCITY:
                ENFORCE_VELOCITY_LOW(2, bc->bottom);
                break;
            default:
                break;
        }
    }
    
    if(c->flags & cell_boundary_top && p->x[2] >= c->dim[2]) {
        switch (bc->top.kind) {
            case BOUNDARY_FREESLIP:
                ENFORCE_FREESLIP_HIGH(2);
                break;
            case BOUNDARY_VELOCITY:
                ENFORCE_VELOCITY_HIGH(2, bc->top);
                break;
            default:
                break;
        }
    }
    
    return enforced;
};


MX_ALWAYS_INLINE bool boundary_potential_eval_ex(std::normal_distribution<float> &gaussian, std::mt19937 &gen,
                            MxPotential *pot, MxParticle *part, MxBoundaryCondition *bc,
                            float *dx, float r2, double *epot) {
    
    float e = 0;
    
    if(pot->kind == POTENTIAL_KIND_DPD) {
        /* update the forces if part in range */
        if (dpd_boundary_eval((DPDPotential*)pot, gaussian(gen), part, bc->velocity.data(), dx, r2 , &e)) {
                        
            /* tabulate the energy */
            *epot += e;
        }
    }
    else {
        float f;
    
        /* update the forces if part in range */
        if (potential_eval_ex(pot, part->radius, bc->radius, r2 , &e , &f )) {
            
            for (int k = 0 ; k < 3 ; k++ ) {
                float w = f * dx[k];
                part->f[k] -= w;
            }
            
            /* tabulate the energy */
            *epot += e;
        }
    }
}


//MX_ALWAYS_INLINE bool potential_eval_super_ex(std::normal_distribution<float> &gaussian, std::mt19937 &gen,
//                            MxPotential *pot, MxParticle *part_i, MxParticle *part_j,
//                            float *dx, float r2, float number_density, double *epot) {

MX_ALWAYS_INLINE bool boundary_eval(std::normal_distribution<float> &gaussian, std::mt19937 &gen,
    MxBoundaryConditions *bc, struct space_cell *cell, MxParticle *part, double *epot ) {
    
    if(!(cell->flags & cell_boundary_any)) {
        return false;
    }
    
    MxPotential *pot;
    float r;
    
    float dx[3] = {0.f, 0.f, 0.f};
    
    
    if((cell->flags & cell_boundary_left) &&
       (pot = bc->left.potenntials[part->typeId]) &&
       ((r = part->x[0]) <= pot->b)) {
        
        dx[0] = -r;
        
        boundary_potential_eval_ex(gaussian, gen, pot, part, &bc->left, dx, r*r, epot);
        
        
    }
    
    
    
}





#endif /* SRC_MDCORE_SRC_BOUNDARY_EVAL_HPP_ */
