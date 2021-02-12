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
#include "space_cell.h"
#include "engine.h"

#include <iostream>

MX_ALWAYS_INLINE bool boundary_update_pos_vel(MxParticle *p, space_cell *c) {
    
    #define ENFORCE_FREESLIP_LOW(i)                              \
        p->position[i] = -p->position[i] * restitution;          \
        p->velocity[i] *= -restitution;                          \
        enforced = true;                                         \

    #define ENFORCE_FREESLIP_HIGH(i)                             \
        p->position[i] = c->dim[i] - (p->position[i] - c->dim[i]) * restitution;          \
        p->velocity[i] *= -restitution;                          \
        enforced = true;                                         \

    
    static const MxBoundaryConditions *bc = &_Engine.boundary_conditions;
    
    static const float restitution = 1.0;
    
    /* Enforce particle position to be within the given boundary */
    bool enforced = false;
    
    if(c->flags & cell_boundary_left && p->x[0] <= 0) {
        switch (bc->left.kind) {
            case BOUNDARY_FREESLIP:
                ENFORCE_FREESLIP_LOW(0);
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
                
            default:
                break;
        }
        
    }
    
    if(c->flags & cell_boundary_front && p->x[1] <= 0) {
        switch (bc->front.kind) {
            case BOUNDARY_FREESLIP:
                ENFORCE_FREESLIP_LOW(1);
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
                
            default:
                break;
        }
    }
    
    if(c->flags & cell_boundary_bottom && p->x[2] <= 0) {
        switch (bc->bottom.kind) {
            case BOUNDARY_FREESLIP:
                ENFORCE_FREESLIP_LOW(2);
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
                
            default:
                break;
        }
    }
    
    return enforced;
};





#endif /* SRC_MDCORE_SRC_BOUNDARY_EVAL_HPP_ */
