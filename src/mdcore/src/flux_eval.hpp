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

MX_ALWAYS_INLINE float flux_fick(MxFlux *flux, int i, float si, float sj) {
    return flux->coef[i] * (si - sj);
}

MX_ALWAYS_INLINE float flux_secrete(MxFlux *flux, int i, float si, float sj) {
    float q = flux->coef[i] * (si - flux->target[i]);
    float scale = q > 0.f;  // forward only, 1 if > 0, 0 if < 0.
    return scale * q;
}

MX_ALWAYS_INLINE float flux_uptake(MxFlux *flux, int i, float si, float sj) {
    float q = flux->coef[i] * (flux->target[i] - sj) * si;
    float scale = q > 0.f;
    return scale * q;
}

MX_ALWAYS_INLINE void flux_eval_ex(
    struct MxFluxes *f , FPTYPE r2 , MxParticle *part_i, MxParticle *part_j ) {
    
    MxFlux *flux = &f->fluxes[0];
    float  r = std::sqrt(r2);
    float term = (1. - r / _Engine.s.cutoff);
    term = term * term;
    
    for(int i = 0; i < flux->size; ++i) {
        // NOTE: order important here, type ids could be the same, i.e.
        // Fick flux, the true branch of each assignemnt gets evaluated.
        MxParticle *pi = part_i->typeId == flux->type_ids[i].a ? part_i : part_j;
        MxParticle *pj = part_j->typeId == flux->type_ids[i].b ? part_j : part_i;
        
        assert(pi->typeId == flux->type_ids[i].a);
        assert(pj->typeId == flux->type_ids[i].b);
        assert(pi != pj);
        
        FPTYPE *si = pi->state_vector->fvec;
        FPTYPE *sj = pj->state_vector->fvec;
        
        FPTYPE *qi = pi->state_vector->q;
        FPTYPE *qj = pj->state_vector->q;
        
        int32_t *ii = flux->indices_a;
        int32_t *ij = flux->indices_b;
        
        float ssi = si[ii[i]];
        float ssj = sj[ij[i]];
        float q =  term;
        
        switch(flux->kinds[i]) {
            case FLUX_FICK:
                q *= flux_fick(flux, i, ssi, ssj);
                break;
            case FLUX_SECRETE:
                q *= flux_secrete(flux, i, ssi, ssj);
                break;
            case FLUX_UPTAKE:
                q *= flux_uptake(flux, i, ssi, ssj);
                break;
            default:
                assert(0);
        }
        
        float half_decay = flux->decay_coef[i] / 2.f;
        qi[ii[i]] = qi[ii[i]] - q - half_decay * ssi;
        qj[ij[i]] = qj[ij[i]] + q - half_decay * ssj;
    }
}

inline MxFluxes *get_fluxes(const MxParticle *a, const MxParticle *b) {
    int index = _Engine.max_type * a->typeId + b->typeId;
    return _Engine.fluxes[index];
}


#endif /* flux_eval_h */
