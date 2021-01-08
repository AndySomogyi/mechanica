/*
 * metrics.cpp
 *
 *  Created on: Nov 18, 2020
 *      Author: andy
 */

#include <metrics.h>
#include <engine.h>
#include <Magnum/Magnum.h>
#include <Magnum/Math/Vector3.h>
#include "MxConvert.hpp"
#include "space.h"
#include "space_cell.h"
#include "runner.h"
#include "MxParticle.h"
#include "potential_eval.h"


static HRESULT virial_pair (float cutoff,
                              const std::set<short int> &typeIds,
                              space_cell *cell_i,
                              space_cell *cell_j,
                              int sid,
                              const Magnum::Vector3 &shift,
                              Magnum::Matrix3 &m);

/**
 * search a pair of cells for particles
 */
static HRESULT enum_particles(const Magnum::Vector3 &origin,
                              float radius,
                              space_cell *cell,
                              const std::set<short int> *typeIds,
                              int32_t exceptPartId,
                              const Magnum::Vector3 &shift,
                              std::vector<int32_t> &ids);


HRESULT MxCalculateVirial(FPTYPE *_origin,
                            FPTYPE radius,
                            const std::set<short int> &typeIds,
                            FPTYPE *tensor) {
    Magnum::Vector3 origin = Magnum::Vector3::from(_origin);
    
    Magnum::Matrix3 m{0.0};
    
    // cell id of target cell
    int cid, ijk[3];
    
    int l[3], ii, jj, kk;
    
    double lh[3];
    
    int id2, sid;
    
    space *s = &_Engine.s;
    
    /** Number of cells within cutoff in each dimension. */
    int span[3];
    
    if((cid = space_get_cellids_for_pos(&_Engine.s, origin.data(), ijk)) < 0) {
        // TODO: bad...
        return E_FAIL;
    }
    
    // the current cell
    space_cell *c = &s->cells[cid];
    
    // the other cell.
    space_cell *cj;
    
    // shift vector between cells.
    Magnum::Vector3 shift;
    
    /* Get the span of the cells we will search for pairs. */
    for (int k = 0 ; k < 3 ; k++ ) {
        span[k] = (int)std::ceil( radius * s->ih[k] );
    }
    
    /* for every neighbouring cell in the x-axis... */
    for ( l[0] = -span[0] ; l[0] <= span[0] ; l[0]++ ) {
        
        /* get coords of neighbour */
        ii = ijk[0] + l[0];
        
        /* wrap or abort if not periodic */
        if (ii < 0 || ii >= s->cdim[0]) {
            continue;
        }

        /* for every neighbouring cell in the y-axis... */
        for ( l[1] = -span[1] ; l[1] <= span[1] ; l[1]++ ) {
            
            /* get coords of neighbour */
            jj = ijk[1] + l[1];
            
            /* wrap or abort if not periodic */
            if ( jj < 0 || jj >= s->cdim[1] ) {
                continue;
            }
            
            /* for every neighbouring cell in the z-axis... */
            for ( l[2] = -span[2] ; l[2] <= span[2] ; l[2]++ ) {
                
                /* get coords of neighbour */
                kk = ijk[2] + l[2];
                
                /* wrap or abort if not periodic */
                if ( kk < 0  ||  kk >= s->cdim[2] ) {
                    continue;
                }
                
                /* Are these cells within the cutoff of each other? */
                lh[0] = s->h[0]*fmax( abs(l[0])-1 , 0 );
                lh[1] = s->h[1]*fmax( abs(l[1])-1 , 0 );
                lh[2] = s->h[2]*fmax( abs(l[2])-1 , 0 );
                if (std::sqrt(lh[0]*lh[0] + lh[1]*lh[1] + lh[2]*lh[2]) > radius )
                    continue;
                
                /* get the neighbour's id */
                id2 = space_cellid(s,ii,jj,kk);
                
                /* Get the pair sortID. */
                c = &s->cells[cid];
                cj = &s->cells[id2];
                sid = space_getsid(s , &c , &cj , shift.data());
                
                HRESULT result = virial_pair (radius, typeIds, c, cj, sid, shift, m);
            } /* for every neighbouring cell in the z-axis... */
        } /* for every neighbouring cell in the y-axis... */
    } /* for every neighbouring cell in the x-axis... */
    
    for(int i = 0; i < 9; ++i) {
        tensor[i] = m.data()[i];
    }

    return S_OK;
}



/**
 * converts cartesian to spherical in global coord space.
 * createsa a numpy array.
 */
PyObject* MPyCartesianToSpherical(const Magnum::Vector3& postion,
                                  const Magnum::Vector3& origin) {
    return mx::cast(MxCartesianToSpherical(postion, origin));
}


/**
 * converts cartesian to spherical, writes spherical
 * coords in to result array.
 */
Magnum::Vector3 MxCartesianToSpherical(const Magnum::Vector3& pos,
                                       const Magnum::Vector3& origin) {
    Magnum::Vector3 vec = pos - origin;
    
    float radius = vec.length();
    float theta = std::atan2(vec.y(), vec.x());
    float phi = std::acos(vec.z() / radius);
    return Magnum::Vector3{radius, theta, phi};
}


static HRESULT virial_pair (float cutoff,
                              const std::set<short int> &typeIds,
                              space_cell *cell_i,
                              space_cell *cell_j,
                              int sid,
                              const Magnum::Vector3 &shift,
                              Magnum::Matrix3 &m) {
    
    int i, j, k, count_i, count_j;
    FPTYPE cutoff2, r2;
    struct MxParticle *part_i, *part_j, *parts_i, *parts_j;
    Magnum::Vector4 dx;
    Magnum::Vector4 pix;
    MxPotential *pot;
    float w = 0, e = 0, f = 0;
    Magnum::Vector3 force;
    

    /* break early if one of the cells is empty */
    count_i = cell_i->count;
    count_j = cell_j->count;
    if ( count_i == 0 || count_j == 0 || ( cell_i == cell_j && count_i < 2 ) )
        return runner_err_ok;
    
    /* get the space and cutoff */
    cutoff2 = cutoff * cutoff;
    pix[3] = FPTYPE_ZERO;
    
    parts_i = cell_i->parts;
    parts_j = cell_j->parts;
    
    /* is this a genuine pair or a cell against itself */
    if ( cell_i == cell_j ) {
        
        /* loop over all particles */
        for ( i = 1 ; i < count_i ; i++ ) {
            
            /* get the particle */
            part_i = &(parts_i[i]);
            pix[0] = part_i->x[0];
            pix[1] = part_i->x[1];
            pix[2] = part_i->x[2];
            
            /* loop over all other particles */
            for ( j = 0 ; j < i ; j++ ) {
                
                /* get the other particle */
                part_j = &(parts_i[j]);
                
                /* get the distance between both particles */
                r2 = fptype_r2(pix.data(), part_j->x , dx.data() );
                
                /* is this within cutoff? */
                if ( r2 > cutoff2 )
                    continue;
                /* runner_rcount += 1; */
                
                /* fetch the potential, if any */
                pot = get_potential(part_i, part_j);
                if ( pot == NULL )
                    continue;
                
                /* check if this is a valid particle to search for */
                if(typeIds.find(part_i->typeId) == typeIds.end() ||
                   typeIds.find(part_j->typeId) == typeIds.end()) {
                    continue;
                }
                
                force[0] = 0; force[1] = 0; force[1] = 0;
                
                /* evaluate the interaction */
                /* update the forces if part in range */
                if (potential_eval_ex(pot, part_i->radius, part_j->radius, r2 , &e , &f )) {
                    for ( k = 0 ; k < 3 ; k++ ) {
                        // divide by two because potential_eval gives double the force
                        // to split beteen a pair of particles.
                        w = (f * dx[k]) / 2;
                        force[k] += w;
                    }
                }
                
                //std::cout << "particle(" << part_i->id << ", " << part_j->id << "), dx:["
                //<< dx[0]    << ", " << dx[1]    << ", " << dx[2]    << "], f:["
                //<< force[0] << ", " << force[1] << ", " << force[2] << "]" << std::endl;
                
                m[0][0] += force[0] * dx[0];
                m[0][1] += force[0] * dx[1];
                m[0][2] += force[0] * dx[2];
                m[1][0] += force[1] * dx[0];
                m[1][1] += force[1] * dx[1];
                m[1][2] += force[1] * dx[2];
                m[2][0] += force[2] * dx[0];
                m[2][1] += force[2] * dx[1];
                m[2][2] += force[2] * dx[2];
            } /* loop over all other particles */
        } /* loop over all particles */
    }
    
    /* no, it's a genuine pair */
    else {
        
        /* loop over all particles */
        for ( i = 0 ; i < count_i ; i++ ) {
            
            // get the particle
            // first particle in in cell_i frame, subtract off shift
            // vector to compute pix in cell_j frame
             
            part_i = &(parts_i[i]);
            pix[0] = part_i->x[0] - shift[0];
            pix[1] = part_i->x[1] - shift[1];
            pix[2] = part_i->x[2] - shift[2];
            
            /* loop over all other particles */
            for ( j = 0 ; j < count_j ; j++ ) {
                
                /* get the other particle */
                part_j = &(parts_j[j]);
                
                /* fetch the potential, if any */
                /* get the distance between both particles */
                r2 = fptype_r2(pix.data() , part_j->x , dx.data() );
                
                /* is this within cutoff? */
                if ( r2 > cutoff2 )
                    continue;
                
                /* fetch the potential, if any */
                pot = get_potential(part_i, part_j);
                if ( pot == NULL )
                    continue;
                
                force[0] = 0; force[1] = 0; force[1] = 0;
                
                /* evaluate the interaction */
                /* update the forces if part in range */
                if (potential_eval_ex(pot, part_i->radius, part_j->radius, r2 , &e , &f )) {
                    for ( k = 0 ; k < 3 ; k++ ) {
                        w = (f * dx[k]) / 2;
                        force[k] += w;
                    }
                }
                
                m[0][0] += force[0] * dx[0];
                m[0][1] += force[0] * dx[1];
                m[0][2] += force[0] * dx[2];
                m[1][0] += force[1] * dx[0];
                m[1][1] += force[1] * dx[1];
                m[1][2] += force[1] * dx[2];
                m[2][0] += force[2] * dx[0];
                m[2][1] += force[2] * dx[1];
                m[2][2] += force[2] * dx[2];
            } /* loop over all other particles */
        } /* loop over all particles */
    }
    
    /* all is well that ends ok */
    return runner_err_ok;
}

HRESULT MxParticles_RadiusOfGyration(int32_t *parts, uint16_t nr_parts,
        float *result)
{
    Magnum::Vector3 r, dx;
    
    float r2 = 0;

    // center of geometry
    for(int i = 0; i < nr_parts; ++i) {
        MxParticle *p = _Engine.s.partlist[parts[i]];
        // global position
        double *o = _Engine.s.celllist[p->id]->origin;
        r[0] += p->x[0] + o[0];
        r[1] += p->x[1] + o[1];
        r[2] += p->x[2] + o[2];
    }
    r = r / nr_parts;
    
    // radial distance squared
    for(int i = 0; i < nr_parts; ++i) {
        MxParticle *p = _Engine.s.partlist[parts[i]];
        // global position
        double *o = _Engine.s.celllist[p->id]->origin;
        
        dx[0] = r[0] - (p->x[0] + o[0]);
        dx[1] = r[1] - (p->x[1] + o[1]);
        dx[2] = r[2] - (p->x[2] + o[2]);
        
        r2 += dx.length();
    }
    
    *result = std::sqrt(r2);
    
    return S_OK;
}

HRESULT MxParticles_CenterOfMass(int32_t *parts, uint16_t nr_parts,
        float *result)
{
    Magnum::Vector3 r;
    float m = 0;
    
    // center of geometry
    for(int i = 0; i < nr_parts; ++i) {
        MxParticle *p = _Engine.s.partlist[parts[i]];
        // global position
        double *o = _Engine.s.celllist[p->id]->origin;
        m += p->mass;
        r[0] += p->mass * (p->x[0] + o[0]);
        r[1] += p->mass * (p->x[1] + o[1]);
        r[2] += p->mass * (p->x[2] + o[2]);
    }
    r = r / m;
    
    result[0] = r[0];
    result[1] = r[1];
    result[2] = r[2];
    
    return S_OK;
}

HRESULT MxParticles_CenterOfGeometry(int32_t *parts, uint16_t nr_parts,
        float *result)
{
    Magnum::Vector3 r;
    
    // center of geometry
    for(int i = 0; i < nr_parts; ++i) {
        MxParticle *p = _Engine.s.partlist[parts[i]];
        // global position
        double *o = _Engine.s.celllist[p->id]->origin;
        r[0] += p->x[0] + o[0];
        r[1] += p->x[1] + o[1];
        r[2] += p->x[2] + o[2];
    }
    r = r / nr_parts;
    
    result[0] = r[0];
    result[1] = r[1];
    result[2] = r[2];
    
    return S_OK;
}

HRESULT MxParticles_MomentOfInertia(int32_t *parts, uint16_t nr_parts,
        float *tensor)
{
    Magnum::Matrix3 m{0.0};
    int i;
    struct MxParticle *part_i;
    Magnum::Vector3 dx;
    Magnum::Vector3 pix;
    Magnum::Vector3 cm;
    HRESULT result = MxParticles_CenterOfMass(parts, nr_parts,cm.data());
    
    if(FAILED(result)) {
        return result;
    }
    
    /* get the space and cutoff */
    pix[3] = FPTYPE_ZERO;
    
    /* loop over all particles */
    for ( i = 0 ; i < nr_parts ; i++ ) {
        
        /* get the particle */
        part_i = _Engine.s.partlist[parts[i]];
        
        // global position of particle i
        double *oi = _Engine.s.celllist[part_i->id]->origin;
        pix[0] = part_i->x[0] + oi[0];
        pix[1] = part_i->x[1] + oi[1];
        pix[2] = part_i->x[2] + oi[2];
        
        // position in center of mass frame
        dx = pix - cm;
        
        m[0][0] += (dx[1]*dx[1] + dx[2]*dx[2]) * part_i->mass;
        m[1][1] += (dx[0]*dx[0] + dx[2]*dx[2]) * part_i->mass;
        m[2][2] += (dx[1]*dx[1] + dx[0]*dx[0]) * part_i->mass;
        m[0][1] += dx[0] * dx[1] * part_i->mass;
        m[1][2] += dx[1] * dx[2] * part_i->mass;
        m[0][2] += dx[0] * dx[2] * part_i->mass;
       
    } /* loop over all particles */
    
    m[1][0] = m[0][1];
    m[2][1] = m[1][2];
    m[2][0] = m[0][2];
    
    for(int i = 0; i < 9; ++i) {
        tensor[i] = m.data()[i];
    }
    
    return S_OK;
}



CAPI_FUNC(HRESULT) MxParticles_Virial(int32_t *parts,
                                                   uint16_t nr_parts,
                                                   uint32_t flags,
                                                   FPTYPE *tensor) {
    Magnum::Matrix3 m{0.0};
    int i, j, k;
    struct MxParticle *part_i, *part_j;
    Magnum::Vector4 dx;
    Magnum::Vector4 pix, pjx;
    MxPotential *pot;
    float w = 0, e = 0, f = 0;
    Magnum::Vector3 force;
    
    /* get the space and cutoff */
    pix[3] = FPTYPE_ZERO;
    
    float r2;
    
    // TODO: more effecient to caclulate everythign in reference frame
    // of outer particle.
        
    /* loop over all particles */
    for ( i = 1 ; i < nr_parts ; i++ ) {
        
        /* get the particle */
        part_i = _Engine.s.partlist[parts[i]];
        
        // global position
        double *oi = _Engine.s.celllist[part_i->id]->origin;
        pix[0] = part_i->x[0] + oi[0];
        pix[1] = part_i->x[1] + oi[1];
        pix[2] = part_i->x[2] + oi[2];
        
        /* loop over all other particles */
        for ( j = 0 ; j < i ; j++ ) {
            
            /* get the other particle */
            part_j = _Engine.s.partlist[parts[j]];
            
            // global position
            double *oj = _Engine.s.celllist[part_j->id]->origin;
            pjx[0] = part_j->x[0] + oj[0];
            pjx[1] = part_j->x[1] + oj[1];
            pjx[2] = part_j->x[2] + oj[2];
            
            /* get the distance between both particles */
            r2 = fptype_r2(pix.data(), pjx.data() , dx.data());
            
            /* fetch the potential, if any */
            pot = get_potential(part_i, part_j);
            if ( pot == NULL )
                continue;
            
            force[0] = 0; force[1] = 0; force[1] = 0;
            
            /* evaluate the interaction */
            /* update the forces if part in range */
            if (potential_eval_ex(pot, part_i->radius, part_j->radius, r2 , &e , &f )) {
                for ( k = 0 ; k < 3 ; k++ ) {
                    // divide by two because potential_eval gives double the force
                    // to split beteen a pair of particles.
                    w = (f * dx[k]) / 2;
                    force[k] += w;
                }
            }
            
            //std::cout << "particle(" << part_i->id << ", " << part_j->id << "), dx:["
            //<< dx[0]    << ", " << dx[1]    << ", " << dx[2]    << "], f:["
            //<< force[0] << ", " << force[1] << ", " << force[2] << "]" << std::endl;
            
            m[0][0] += force[0] * dx[0];
            m[0][1] += force[0] * dx[1];
            m[0][2] += force[0] * dx[2];
            m[1][0] += force[1] * dx[0];
            m[1][1] += force[1] * dx[1];
            m[1][2] += force[1] * dx[2];
            m[2][0] += force[2] * dx[0];
            m[2][1] += force[2] * dx[1];
            m[2][2] += force[2] * dx[2];
        } /* loop over all other particles */
    } /* loop over all particles */
    
    for(int i = 0; i < 9; ++i) {
        tensor[i] = m.data()[i];
    }
    
    return S_OK;
}


HRESULT MxParticle_Neighbors(MxParticle *part,
                               FPTYPE radius,
                               const std::set<short int> *typeIds,
                               uint16_t *nr_parts,
                               int32_t **pparts)  {
    // origin in global space
    Magnum::Vector3 origin = part->global_position();
    
    // cell id of target cell
    int cid, ijk[3];
    
    int l[3], ii, jj, kk;
    
    double lh[3];
    
    int id2, sid;
    
    space *s = &_Engine.s;
    
    /** Number of cells within cutoff in each dimension. */
    int span[3];
    
    std::vector<int32_t> ids;
    
    if((cid = space_get_cellids_for_pos(&_Engine.s, origin.data(), ijk)) < 0) {
        // TODO: bad...
        return E_FAIL;
    }
    
    // std::cout << "origin cell: " << cid << "(" << ijk[0] << "," << ijk[1] << "," << ijk[2] << ")" << std::endl;
    
    // the current cell
    space_cell *c = &s->cells[cid];
    
    // origin in the target cell's coordinate system
    Magnum::Vector3 local_origin = {
        (float)(origin[0] - c->origin[0]),
        (float)(origin[1] - c->origin[1]),
        (float)(origin[2] - c->origin[2])
    };
    
    // the other cell.
    space_cell *cj, *ci;
    
    // shift vector between cells.
    Magnum::Vector3 shift;
    
    /* Get the span of the cells we will search for pairs. */
    for (int k = 0 ; k < 3 ; k++ ) {
        span[k] = (int)std::ceil( radius * s->ih[k] );
    }
    
    /* for every neighbouring cell in the x-axis... */
    for ( l[0] = -span[0] ; l[0] <= span[0] ; l[0]++ ) {
        
        /* get coords of neighbour */
        ii = ijk[0] + l[0];
        
        /* wrap or abort if not periodic */
        if ( ii < 0 ) {
            if (s->period & space_periodic_x)
                ii += s->cdim[0];
            else
                continue;
        }
        else if ( ii >= s->cdim[0] ) {
            if (s->period & space_periodic_x)
                ii -= s->cdim[0];
            else
                continue;
        }
        
        /* for every neighbouring cell in the y-axis... */
        for ( l[1] = -span[1] ; l[1] <= span[1] ; l[1]++ ) {
            
            /* get coords of neighbour */
            jj = ijk[1] + l[1];
            
            /* wrap or abort if not periodic */
            if ( jj < 0 ) {
                if (s->period & space_periodic_y)
                    jj += s->cdim[1];
                else
                    continue;
            }
            else if ( jj >= s->cdim[1] ) {
                if (s->period & space_periodic_y)
                    jj -= s->cdim[1];
                else
                    continue;
            }
            
            /* for every neighbouring cell in the z-axis... */
            for ( l[2] = -span[2] ; l[2] <= span[2] ; l[2]++ ) {
                
                /* get coords of neighbour */
                kk = ijk[2] + l[2];
                
                /* wrap or abort if not periodic */
                if ( kk < 0 ) {
                    if (s->period & space_periodic_z)
                        kk += s->cdim[2];
                    else
                        continue;
                }
                else if ( kk >= s->cdim[2] ) {
                    if (s->period & space_periodic_z)
                        kk -= s->cdim[2];
                    else
                        continue;
                }
                
                /* Are these cells within the cutoff of each other? */
                lh[0] = s->h[0]*fmax( abs(l[0])-1 , 0 );
                lh[1] = s->h[1]*fmax( abs(l[1])-1 , 0 );
                lh[2] = s->h[2]*fmax( abs(l[2])-1 , 0 );
                if (std::sqrt(lh[0]*lh[0] + lh[1]*lh[1] + lh[2]*lh[2]) > radius )
                    continue;
                
                /* get the neighbour's id */
                id2 = space_cellid(s,ii,jj,kk);
                
                /* Get the pair sortID. */
                ci = &s->cells[cid];
                cj = &s->cells[id2];
                sid = space_getsid(s , &ci , &cj , shift.data());
                
                // check if flipped,
                // space_getsid flips cells under certain circumstances.
                if(cj == c) {
                    cj = ci;
                    shift = shift * -1;
                }
                
                //std::cout << id2 << ":(" << ii << "," << jj << "," << kk << "), ("
                // << shift[0] << ", " << shift[1] << ", " << shift[2] << ")" << std::endl;
                
                HRESULT result = enum_particles (local_origin, radius, cj, typeIds, part->id, shift, ids);
            } /* for every neighbouring cell in the z-axis... */
        } /* for every neighbouring cell in the y-axis... */
    } /* for every neighbouring cell in the x-axis... */
    
    *nr_parts = ids.size();
    int32_t *parts = (int32_t*)malloc(ids.size() * sizeof(int32_t));
    memcpy(parts, ids.data(), ids.size() * sizeof(int32_t));
    *pparts = parts;
    
    return S_OK;
}


HRESULT enum_particles(const Magnum::Vector3 &_origin,
                       float radius,
                       space_cell *cell,
                       const std::set<short int> *typeIds,
                       int32_t exceptPartId,
                       const Magnum::Vector3 &shift,
                       std::vector<int32_t> &ids) {
    
    int i, count;
    FPTYPE cutoff2, r2;
    struct MxParticle *part, *parts;
    Magnum::Vector4 dx;
    Magnum::Vector4 pix;
    Magnum::Vector4 origin;
    
    /* break early if one of the cells is empty */
    count = cell->count;
    
    if ( count == 0 )
        return runner_err_ok;
    
    /* get the space and cutoff */
    cutoff2 = radius * radius;
    pix[3] = FPTYPE_ZERO;
    
    parts = cell->parts;
    
    // shift the origin into the current cell's reference
    // frame with the shift vector.
    origin[0] = _origin[0] - shift[0];
    origin[1] = _origin[1] - shift[1];
    origin[2] = _origin[2] - shift[2];
    
    /* loop over all other particles */
    for ( i = 0 ; i < count ; i++ ) {
        
        /* get the other particle */
        part = &(parts[i]);
        
        if(part->id == exceptPartId) {
            continue;
        }
        
        /* get the distance between both particles */
        r2 = fptype_r2(origin.data() , part->x , dx.data() );
        
        /* is this within cutoff? */
        if ( r2 > cutoff2 ) {
            continue;
        }
        
        /* check if this is a valid particle to search for */
        if(typeIds && typeIds->find(part->typeId) == typeIds->end()) {
            continue;
        }
        
        ids.push_back(part->id);
        
    } /* loop over all other particles */
    
    
    /* all is well that ends ok */
    return runner_err_ok;
}


