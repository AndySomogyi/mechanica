//
//  flux_eval.hpp
//  Mechanica
//
//  Created by Andy on 12/30/20.
//
#pragma once
#ifndef flux_eval_h
#define flux_eval_h


#include "Flux.hpp"
#include "CStateVector.hpp"
#include "engine.h"
#include <iostream>


/* This file contains the potential evaluation function als "extern inline",
 such that they can be inlined in the respective modules.
 
 If your code wants to call any potential_eval functions, you must include
 this file.
 */

/* Function prototypes. */
/* void potential_eval ( struct potential *p , FPTYPE r2 , FPTYPE *e , FPTYPE *f );
 void potential_eval_expl ( struct potential *p , FPTYPE r2 , FPTYPE *e , FPTYPE *f );
 void potential_eval_vec_4single ( struct potential *p[4] , float *r2 , float *e , float *f );
 void potential_eval_vec_4single_r ( struct potential *p[4] , float *r_in , float *e , float *f );
 void potential_eval_vec_8single ( struct potential *p[4] , float *r2 , float *e , float *f );
 void potential_eval_vec_2double ( struct potential *p[4] , FPTYPE *r2 , FPTYPE *e , FPTYPE *f );
 void potential_eval_vec_4double ( struct potential *p[4] , FPTYPE *r2 , FPTYPE *e , FPTYPE *f );
 void potential_eval_vec_4double_r ( struct potential *p[4] , FPTYPE *r , FPTYPE *e , FPTYPE *f );
 void potential_eval_r ( struct potential *p , FPTYPE r , FPTYPE *e , FPTYPE *f );
 */


/* Get the inlining right. */
#ifndef INLINE
# if __GNUC__ && !__GNUC_STDC_INLINE__
#  define INLINE extern inline
# else
#  define INLINE inline
# endif
#endif



__attribute__ ((always_inline)) INLINE void flux_eval_ex(
    struct MxFluxes *f , FPTYPE r2 , MxParticle *part_i, MxParticle *part_j ) {
    
    FPTYPE *si = part_i->state_vector->fvec;
    FPTYPE *sj = part_j->state_vector->fvec;
    
    FPTYPE *qi = part_i->state_vector->q;
    FPTYPE *qj = part_j->state_vector->q;
    
    int32_t *ii = f->indices_a;
    int32_t *ij = f->indices_b;
    
    float  r = std::sqrt(r2);
    float term = (1. - r / _Engine.s.cutoff);
    term = term * term;
    
    for(int i = 0; i < f->size; ++i) {
        float ssi = si[ii[i]];
        float ssj = sj[ij[i]];
        float q = f->coef[i] * term * (ssi - ssj);
        float decay = f->decay_coef[i] * ssi;
        qi[ii[i]] -= (q + decay);
        qj[ij[i]] += q;
    }
}

inline MxFluxes *get_fluxes(const MxParticle *a, const MxParticle *b) {
    int index = _Engine.max_type * a->typeId + b->typeId;
    return _Engine.fluxes[index];
}


#endif /* flux_eval_h */
