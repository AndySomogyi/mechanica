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
        if(bc->left.kind & BOUNDARY_FREESLIP) {
            ENFORCE_FREESLIP_LOW(0);
        }
        else if(bc->left.kind & BOUNDARY_VELOCITY) {
            ENFORCE_VELOCITY_LOW(0, bc->left);
        }
    }

    if(c->flags & cell_boundary_right && p->x[0] >= c->dim[0]) {
        if(bc->right.kind & BOUNDARY_FREESLIP) {
            ENFORCE_FREESLIP_HIGH(0);
        }
        else if(bc->right.kind & BOUNDARY_VELOCITY) {
            ENFORCE_VELOCITY_HIGH(0, bc->right);
        }
    }

    if(c->flags & cell_boundary_front && p->x[1] <= 0) {
        if(bc->front.kind & BOUNDARY_FREESLIP) {
            ENFORCE_FREESLIP_LOW(1);
        }
        else if(bc->front.kind & BOUNDARY_VELOCITY) {
            ENFORCE_VELOCITY_LOW(1, bc->front);
        }
    }

    if(c->flags & cell_boundary_back && p->x[1] >= c->dim[1]) {
        if(bc->back.kind & BOUNDARY_FREESLIP) {
            ENFORCE_FREESLIP_HIGH(1);
        }
        else if(bc->back.kind & BOUNDARY_VELOCITY) {
            ENFORCE_VELOCITY_HIGH(1, bc->back);
        }
    }

    if(c->flags & cell_boundary_bottom && p->x[2] <= 0) {
        if(bc->bottom.kind & BOUNDARY_FREESLIP) {
            ENFORCE_FREESLIP_LOW(2);
        }
        else if(bc->bottom.kind & BOUNDARY_VELOCITY) {
            ENFORCE_VELOCITY_LOW(2, bc->bottom);
        }
    }

    if(c->flags & cell_boundary_top && p->x[2] >= c->dim[2]) {
        if(bc->top.kind & BOUNDARY_FREESLIP) {
            ENFORCE_FREESLIP_HIGH(2);
        }
        else if(bc->top.kind & BOUNDARY_VELOCITY) {
            ENFORCE_VELOCITY_HIGH(2, bc->top);
        }
    }

    return enforced;
};


MX_ALWAYS_INLINE bool boundary_potential_eval_ex(const struct space_cell *cell,
                            MxPotential *pot, MxParticle *part, MxBoundaryCondition *bc,
                            float *dx, float r2, double *epot) {
    float e = 0;
    bool result = false;

    if(pot->kind == POTENTIAL_KIND_DPD) {
        /* update the forces if part in range */
        if (dpd_boundary_eval((DPDPotential*)pot, space_cell_gaussian(cell->id), part, bc->velocity.data(), dx, r2 , &e)) {
            /* tabulate the energy */
            *epot += e;
            result = true;
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
            result = true;
        }
    }
    return result;
}


//MX_ALWAYS_INLINE bool potential_eval_super_ex(std::normal_distribution<float> &gaussian, std::mt19937 &gen,
//                            MxPotential *pot, MxParticle *part_i, MxParticle *part_j,
//                            float *dx, float r2, float number_density, double *epot) {

MX_ALWAYS_INLINE bool boundary_eval(
    MxBoundaryConditions *bc, const struct space_cell *cell, MxParticle *part, double *epot ) {
    
    MxPotential *pot;
    float r;
    bool result = false;
    
    float dx[3] = {0.f, 0.f, 0.f};
        
    if((cell->flags & cell_boundary_left) &&
       (pot = bc->left.potenntials[part->typeId]) &&
       ((r = part->x[0]) <= pot->b)) {
        dx[0] = r;
        result |= boundary_potential_eval_ex(cell, pot, part, &bc->left, dx, r*r, epot);
    }
    
    if((cell->flags & cell_boundary_right) &&
       (pot = bc->right.potenntials[part->typeId]) &&
       ((r = cell->dim[0] - part->x[0]) <= pot->b)) {
        dx[0] = -r;
        result |= boundary_potential_eval_ex(cell, pot, part, &bc->right, dx, r*r, epot);
    }
    
    if((cell->flags & cell_boundary_front) &&
       (pot = bc->front.potenntials[part->typeId]) &&
       ((r = part->x[1]) <= pot->b)) {
        dx[1] = r;
        result |= boundary_potential_eval_ex(cell, pot, part, &bc->front, dx, r*r, epot);
    }
    
    if((cell->flags & cell_boundary_back) &&
       (pot = bc->back.potenntials[part->typeId]) &&
       ((r = cell->dim[1] - part->x[1]) <= pot->b)) {
        dx[1] = -r;
        result |= boundary_potential_eval_ex(cell, pot, part, &bc->back, dx, r*r, epot);
    }
    
    if((cell->flags & cell_boundary_bottom) &&
       (pot = bc->bottom.potenntials[part->typeId]) &&
       ((r = part->x[2]) <= pot->b)) {
        dx[2] = r;
        result |= boundary_potential_eval_ex(cell, pot, part, &bc->bottom, dx, r*r, epot);
    }
    
    if((cell->flags & cell_boundary_top) &&
       (pot = bc->top.potenntials[part->typeId]) &&
       ((r = cell->dim[2] - part->x[2]) <= pot->b)) {
        dx[2] = -r;
        result |= boundary_potential_eval_ex(cell, pot, part, &bc->top, dx, r*r, epot);
    }
    return result;
}




#endif /* SRC_MDCORE_SRC_BOUNDARY_EVAL_HPP_ */
