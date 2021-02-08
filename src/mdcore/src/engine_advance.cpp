/*
 * engine_advance.cpp
 *
 *  Created on: Jan 2, 2021
 *      Author: andy
 */

/* Include conditional headers. */
#include "mdcore_config.h"
#include "engine.h"
#include "engine_advance.h"
#include "errs.h"
#include "MxCluster.hpp"
#include "Flux.hpp"

#include <sstream>
#pragma clang diagnostic ignored "-Wwritable-strings"
#include <iostream>


#if MX_THREADING
#include "MxTaskScheduler.hpp"
#endif


#ifdef WITH_MPI
#include <mpi.h>
#endif
#ifdef HAVE_OPENMP
#include <omp.h>
#endif
#ifdef WITH_METIS
#include <metis.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#endif

/* the error macro. */
#define error(id) ( engine_err = errs_register( id , engine_err_msg[-(id)] , __LINE__ , __FUNCTION__ , __FILE__ ) )


static std::mutex space_mutex;
static std::mutex tot_energy_mutex;

static int engine_advance_forward_euler ( struct engine *e );
static int engine_advance_runge_kutta_4 ( struct engine *e );
static bool engine_enforce_boundary(engine *e, MxParticle *p, space_cell *c) {
    
#define ENFORCE_BC(i)                                                   \
    if(ppos[i] < 0) {                                                   \
        p->position[i] += (0 - ppos[i])* (restitution + 1.0f);          \
        p->velocity[i] *= -restitution;                                 \
        enforced = true;                                                \
    }                                                                   \
    else if(ppos[i] > e->s.dim[i]) {                                    \
        p->position[i] -= (ppos[i] - e->s.dim[i])*(restitution + 1.0f); \
        p->velocity[i] *= -restitution;                                 \
        enforced = true;                                                \
    }                                                                   \
    
    
    float restitution = 1.0;
    /* Enforce particle position to be within the given boundary */
    bool enforced = false;
    
    if(!(e->s.period & SPACE_FREESLIP_FULL)) {
        return false;
    }
    
    double *o = c->origin;
    Magnum::Vector3 ppos = {
        static_cast<float>(o[0]) + p->position[0],
        static_cast<float>(o[1]) + p->position[1],
        static_cast<float>(o[2]) + p->position[2]
    };
    
    if(e->s.period & SPACE_FREESLIP_X) {
        ENFORCE_BC(0);
    }
    if(e->s.period & SPACE_FREESLIP_Y) {
        ENFORCE_BC(1);
    }
    if(e->s.period & SPACE_FREESLIP_Z) {
        ENFORCE_BC(2);
    }
    
    /*
    for(int i = 0; i != 3; ++i) {
        if(ppos[i] < _lowerDomainBound[i]) {
            ppos[i] += (_lowerDomainBound[i] - ppos[i])*(restitution + 1.0f);
            pvel[i] *= -restitution;
            bVelChanged = true;
        } else if(ppos[i] > _upperDomainBound[i]) {
            ppos[i] -= (ppos[i] - _upperDomainBound[i])*(restitution + 1.0f);
            pvel[i] *= -restitution;
            bVelChanged = true;
        }
    }
     */
    
    return enforced;
};

static int _toofast_error(MxParticle *p, int line, const char* func) {
    //CErr_Set(HRESULT code, const char* msg, int line, const char* file, const char* func);
    std::stringstream ss;
    ss << "ERROR, particle moving too fast, p: {" << std::endl;
    ss << "\tid: " << p->id << ", " << std::endl;
    ss << "\ttype: " << _Engine.types[p->typeId].name << "," << std::endl;
    ss << "\tx: [" << p->x[0] << ", " << p->x[1] << ", " << p->x[2] << "], " << std::endl;
    ss << "\tv: [" << p->v[0] << ", " << p->v[1] << ", " << p->v[2] << "], " << std::endl;
    ss << "\tf: [" << p->f[0] << ", " << p->f[1] << ", " << p->f[2] << "], " << std::endl;
    ss << "}";
    
    CErr_Set(E_FAIL, ss.str().c_str(), line, __FILE__, func);
    return error(engine_err_toofast);
}

#define toofast_error(p) _toofast_error(p, __LINE__, MX_FUNCTION)


/**
 * @brief Update the particle velocities and positions, re-shuffle if
 *      appropriate.
 * @param e The #engine on which to run.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */

int engine_advance ( struct engine *e ) {
    if(e->integrator == EngineIntegrator::FORWARD_EULER) {
        return engine_advance_forward_euler(e);
    }
    else {
        return engine_advance_runge_kutta_4(e);
    }
}

Magnum::Quaternion integrate_angular_velocity_exact_1(const Magnum::Vector3& em, double deltaTime)
{
    Magnum::Vector3 ha = em * deltaTime * 0.5; // vector of half angle
    double len = ha.length(); // magnitude
    if (len > 0) {
        ha *= std::sin(len) / len;
        double w = std::cos(len);
        double s = std::sin(len) / len;
        return Magnum::Quaternion(ha, w);
    } else {
        return Magnum::Quaternion(ha, 1.0);
    }
}

static Magnum::Quaternion integrate_angular_velocity_2(const Magnum::Vector3 &av, double dt) {
    float len = av.length();
    double theta = len * dt * 0.5;
    if (len > 1.0e-12) {
        double w = std::cos(theta);
        double s = std::sin(theta) / len;
        return  Magnum::Quaternion(av * s, w);
    } else {
        return Magnum::Quaternion({0.f, 0.f, 0.f}, 1.f);
    }
}

static inline void bodies_advance_forward_euler(const float dt, int cid)
{
    if(cid == 0) {
        for (MxCuboid& c : _Engine.s.cuboids) {
            c.orientation = c.orientation * integrate_angular_velocity_2(c.spin, dt);
            MxCuboid_UpdateAABB(&c);
        }
    }
}


// FPTYPE dt, h[3], h2[3], maxv[3], maxv2[3], maxx[3], maxx2[3]; // h, h2: edge length of space cells.

static inline void cell_advance_forward_euler(const float dt, const float h[3], const float h2[3],
                   const float maxv[3], const float maxv2[3], const float maxx[3],
                   const float maxx2[3], int cid)
{
    space *s = &_Engine.s;
    struct space_cell *c, *c_dest;
    int pid = 0;
    MxParticle *p;
    int toofast;
    float dx, v, neg;
    int k;
    int delta[3];
    
    c = &(s->cells[ s->cid_real[cid] ]);
    
    tot_energy_mutex.lock();
    s->epot += c->epot;
    tot_energy_mutex.unlock();
    
    while ( pid < c->count ) {
        p = &( c->parts[pid] );
        toofast = 0;
        
        if(p->flags & PARTICLE_CLUSTER || (
                                           (p->flags & PARTICLE_FROZEN_X) &&
                                           (p->flags & PARTICLE_FROZEN_Y) &&
                                           (p->flags & PARTICLE_FROZEN_Z)
                                           )) {
            pid++;
            continue;
        }
        
        float mask[] = {
            (p->flags & PARTICLE_FROZEN_X) ? 0.0f : 1.0f,
            (p->flags & PARTICLE_FROZEN_Y) ? 0.0f : 1.0f,
            (p->flags & PARTICLE_FROZEN_Z) ? 0.0f : 1.0f
        };
        
        if(engine::types[p->typeId].dynamics == PARTICLE_NEWTONIAN) {
            for ( k = 0 ; k < 3 ; k++ ) {
                v = mask[k] * (p->v[k] +  dt * p->f[k] * p->imass);
                neg = v / abs(v);
                p->v[k] = v * v <= maxv2[k] ? v : neg * maxv[k];
                p->x[k] += dt * p->v[k];
                delta[k] = __builtin_isgreaterequal( p->x[k] , h[k] ) - __builtin_isless( p->x[k] , 0.0 );
                toofast = toofast || (p->x[k] >= h2[k] || p->x[k] <= -h[k]);
            }
        }
        else {
            for ( k = 0 ; k < 3 ; k++ ) {
                dx = mask[k] * (dt * p->f[k] * p->imass);
                neg = dx / abs(dx); // could be NaN, but only used if dx is > maxx.
                p->x[k] += dx * dx <= maxx2[k] ? dx : neg * maxx[k];
                delta[k] = __builtin_isgreaterequal( p->x[k] , h[k] ) - __builtin_isless( p->x[k] , 0.0 );
                toofast = toofast || (p->x[k] >= h2[k] || p->x[k] <= -h[k]);
            }
        }
        
        p->inv_number_density = p->number_density > 0.f ? 1.f / p->number_density : 0.f;
        _Engine.computed_volume += p->inv_number_density;
        
        if(toofast) {
            toofast_error(p);
        }
        
        /* do we have to move this particle? */
        // TODO: consolidate moving to one method.
        
        // if delta is non-zero, need to check boundary conditions, and
        // if moved out of cell, or out of bounds.
        if ( ( delta[0] != 0 ) || ( delta[1] != 0 ) || ( delta[2] != 0 ) ) {
            
            // if we enforce boundary, reflect back into same cell
            if(engine_enforce_boundary(&_Engine, p, c)) {
                pid += 1;
            }
            // otherwise move to different cell
            else {
                for ( k = 0 ; k < 3 ; k++ ) {
                    p->x[k] -= delta[k] * h[k];
                    p->p0[k] -= delta[k] * h[k];
                }
                c_dest = &( s->cells[ space_cellid( s ,
                                                   (c->loc[0] + delta[0] + s->cdim[0]) % s->cdim[0] ,
                                                   (c->loc[1] + delta[1] + s->cdim[1]) % s->cdim[1] ,
                                                   (c->loc[2] + delta[2] + s->cdim[2]) % s->cdim[2] ) ] );
                
                space_mutex.lock();
                space_cell_add_incomming( c_dest , p );
                space_mutex.unlock();
                
                s->celllist[ p->id ] = c_dest;
                
                // remove a particle from a cell. if the part was the last in the
                // cell, simply dec the count, otherwise, move the last part
                // in the cell to the ejected part's prev loc.
                c->count -= 1;
                if ( pid < c->count ) {
                    c->parts[pid] = c->parts[c->count];
                    s->partlist[ c->parts[pid].id ] = &( c->parts[pid] );
                }
            }
        }
        else {
            pid += 1;
        }
    }
}

/**
 * @brief Update the particle velocities and positions, re-shuffle if
 *      appropriate.
 * @param e The #engine on which to run.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
int engine_advance_forward_euler ( struct engine *e ) {

    // set the integrator flag to set any persistent forces
    // forward euler is a single step, so alwasy set this flag
    e->integrator_flags |= INTEGRATOR_UPDATE_PERSISTENTFORCE;

    if (engine_force( e ) < 0 ) {
        return error(engine_err);
    }

    int cid, pid, k, delta[3], step;
    struct space_cell *c, *c_dest;
    struct MxParticle *p;
    struct space *s;
    FPTYPE dt, h[3], h2[3], maxv[3], maxv2[3], maxx[3], maxx2[3]; // h, h2: edge length of space cells.
    double epot = 0.0, epot_local;
    int toofast;
    float dx, v, neg;

    /* Get a grip on the space. */
    s = &(e->s);
    dt = e->dt;
    for ( k = 0 ; k < 3 ; k++ ) {
        h[k] = s->h[k];
        h2[k] = 2. * s->h[k];
        
        // max velocity and step, as a fraction of cell size.
        maxv[k] = (h[k] * e->particle_max_dist_fraction) / dt;
        maxv2[k] = maxv[k] * maxv[k];
        
        maxx[k] = h[k] * e->particle_max_dist_fraction;
        maxx2[k] = maxx[k] * maxx[k];
    }
    
    e->computed_volume = 0;

    /* update the particle velocities and positions */
    if ((e->flags & engine_flag_verlet) || (e->flags & engine_flag_mpi)) {

        /* Collect potential energy from ghosts. */
        for ( cid = 0 ; cid < s->nr_ghost ; cid++ )
            epot += s->cells[ s->cid_ghost[cid] ].epot;

#pragma omp parallel private(cid,c,pid,p,w,k,epot_local)
        {
            step = omp_get_num_threads();
            epot_local = 0.0;
            for ( cid = omp_get_thread_num() ; cid < s->nr_real ; cid += step ) {
                c = &(s->cells[ s->cid_real[cid] ]);
                epot_local += c->epot;
                for ( pid = 0 ; pid < c->count ; pid++ ) {
                    p = &( c->parts[pid] );

                    toofast = 0;
                    if(engine::types[p->typeId].dynamics == PARTICLE_NEWTONIAN) {
                        for ( k = 0 ; k < 3 ; k++ ) {
                            p->v[k] += p->f[k] * dt * p->imass;
                            p->x[k] += p->v[k] * dt;
                            delta[k] = isgreaterequal( p->x[k] , h[k] ) - isless( p->x[k] , 0.0 );
                            toofast = toofast || (p->x[k] >= h2[k] || p->x[k] <= -h[k]);
                        }
                    }
                    else {
                        for ( k = 0 ; k < 3 ; k++ ) {
                            p->x[k] += p->f[k] * dt * p->imass;
                            delta[k] = isgreaterequal( p->x[k] , h[k] ) - isless( p->x[k] , 0.0 );
                            toofast = toofast || (p->x[k] >= h2[k] || p->x[k] <= -h[k]);
                        }
                    }
                }
            }
#pragma omp atomic
            epot += epot_local;
        }
    }
    else { // NOT if ((e->flags & engine_flag_verlet) || (e->flags & engine_flag_mpi)) {

        /* Collect potential energy from ghosts. */
        for ( cid = 0 ; cid < s->nr_ghost ; cid++ ) {
            epot += s->cells[ s->cid_ghost[cid] ].epot;
        }
        
        // make a lambda function that we run in parallel, capture local vars.
        // we use the same lambda in both parallel and serial versions to
        // make sure same code gets exercized.
        //
        // cell_advance_forward_euler(const float dt, const float h[3], const float h2[3],
        // const float maxv[3], const float maxv2[3], const float maxx[3],
        // const float maxx2[3], float *total_pot, int cid)
        
        auto func = [dt, &h, &h2, &maxv, &maxv2, &maxx, &maxx2, &epot](int cid) -> void {
            cell_advance_forward_euler(dt, h, h2, maxv, maxv2, maxx, maxx2, cid);
            
            MxFluxes_Integrate(cid);
            
            bodies_advance_forward_euler(dt, cid);
        };
        
#if MX_THREADING
        mx::parallel_for(s->nr_real, func);
#else
        for(cid = 0; cid < s->nr_real; ++cid) {
            func(cid);
        }
#endif
        
        std::cout << "step: " << _Engine.time  << ", computed volume: " << _Engine.computed_volume << std::endl;
        
        /* set the new pos for the clusters.  */
        for ( cid = 0 ; cid < s->nr_real ; ++cid ) {
            c = &(s->cells[ s->cid_real[cid] ]);
            pid = 0;
            while ( pid < c->count ) {
                p = &( c->parts[pid] );
                if((p->flags & PARTICLE_CLUSTER) && p->nr_parts > 0) {
                    
                    MxCluster_ComputeAggregateQuantities((MxCluster*)p);
                    
                    for ( k = 0 ; k < 3 ; k++ ) {
                        delta[k] = __builtin_isgreaterequal( p->x[k] , h[k] ) - __builtin_isless( p->x[k] , 0.0 );
                    }
                    
                    /* do we have to move this particle? */
                    // TODO: consolidate moving to one method.
                    if ( ( delta[0] != 0 ) || ( delta[1] != 0 ) || ( delta[2] != 0 ) ) {
                        for ( k = 0 ; k < 3 ; k++ ) {
                            p->x[k] -= delta[k] * h[k];
                            p->p0[k] -= delta[k] * h[k];
                        }
                        
                        c_dest = &( s->cells[ space_cellid( s ,
                                                           (c->loc[0] + delta[0] + s->cdim[0]) % s->cdim[0] ,
                                                           (c->loc[1] + delta[1] + s->cdim[1]) % s->cdim[1] ,
                                                           (c->loc[2] + delta[2] + s->cdim[2]) % s->cdim[2] ) ] );
                        
                        pthread_mutex_lock(&c_dest->cell_mutex);
                        space_cell_add_incomming( c_dest , p );
                        pthread_mutex_unlock(&c_dest->cell_mutex);
                        
                        s->celllist[ p->id ] = c_dest;
                        
                        // remove a particle from a cell. if the part was the last in the
                        // cell, simply dec the count, otherwise, move the last part
                        // in the cell to the ejected part's prev loc.
                        c->count -= 1;
                        if ( pid < c->count ) {
                            c->parts[pid] = c->parts[c->count];
                            s->partlist[ c->parts[pid].id ] = &( c->parts[pid] );
                        }
                    }
                    else {
                        pid += 1;
                    }
                }
                else {
                    pid += 1;
                }
            }
        }

        /* Welcome the new particles in each cell. */
        for ( cid = 0 ; cid < s->nr_marked ; cid++ ) {
            space_cell_welcome( &(s->cells[ s->cid_marked[cid] ]) , s->partlist );
        }
    } // endif NOT if ((e->flags & engine_flag_verlet) || (e->flags & engine_flag_mpi))

    /* Store the accumulated potential energy. */
    s->epot_nonbond += epot;
    s->epot += epot;

    /* return quietly */
    return engine_err_ok;
}

#define CHECK_TOOFAST(p, h, h2) \
{\
    for(int _k = 0; _k < 3; _k++) {\
        if (p->x[_k] >= h2[_k] || p->x[_k] <= -h[_k]) {\
            return toofast_error(p);\
        }\
    }\
}\



/**
 * @brief Update the particle velocities and positions, re-shuffle if
 *      appropriate.
 * @param e The #engine on which to run.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */

int engine_advance_runge_kutta_4 ( struct engine *e ) {

    int cid, pid, k, delta[3], step;
    struct space_cell *c, *c_dest;
    struct MxParticle *p;
    struct space *s;
    FPTYPE dt, w, h[3], h2[3], maxv[3], maxv2[3], maxx[3], maxx2[3]; // h, h2: edge length of space cells.
    double epot = 0.0, epot_local;
    int toofast;

    /* Get a grip on the space. */
    s = &(e->s);
    dt = e->dt;
    for ( k = 0 ; k < 3 ; k++ ) {
        h[k] = s->h[k];
        h2[k] = 2. * s->h[k];

        maxv[k] = h[k] / (e->particle_max_dist_fraction * dt);
        maxv2[k] = maxv[k] * maxv[k];

        maxx[k] = h[k] / (e->particle_max_dist_fraction);
        maxx2[k] = maxx[k] * maxx[k];
    }

    /* update the particle velocities and positions */
    if ((e->flags & engine_flag_verlet) || (e->flags & engine_flag_mpi)) {

        /* Collect potential energy from ghosts. */
        for ( cid = 0 ; cid < s->nr_ghost ; cid++ )
            epot += s->cells[ s->cid_ghost[cid] ].epot;

#pragma omp parallel private(cid,c,pid,p,w,k,epot_local)
        {
            step = omp_get_num_threads();
            epot_local = 0.0;
            for ( cid = omp_get_thread_num() ; cid < s->nr_real ; cid += step ) {
                c = &(s->cells[ s->cid_real[cid] ]);
                epot_local += c->epot;
                for ( pid = 0 ; pid < c->count ; pid++ ) {
                    p = &( c->parts[pid] );
                    w = dt * p->imass;

                    toofast = 0;
                    if(engine::types[p->typeId].dynamics == PARTICLE_NEWTONIAN) {
                        for ( k = 0 ; k < 3 ; k++ ) {
                            p->v[k] += dt * p->f[k] * w;
                            p->x[k] += dt * p->v[k];
                            delta[k] = isgreaterequal( p->x[k] , h[k] ) - isless( p->x[k] , 0.0 );
                            toofast = toofast || (p->x[k] >= h2[k] || p->x[k] <= -h[k]);
                        }
                    }
                    else {
                        for ( k = 0 ; k < 3 ; k++ ) {
                            p->x[k] += dt * p->f[k] * w;
                            delta[k] = isgreaterequal( p->x[k] , h[k] ) - isless( p->x[k] , 0.0 );
                            toofast = toofast || (p->x[k] >= h2[k] || p->x[k] <= -h[k]);
                        }
                    }
                }
            }
#pragma omp atomic
            epot += epot_local;
        }
    }
    else { // NOT if ((e->flags & engine_flag_verlet) || (e->flags & engine_flag_mpi))

        /* Collect potential energy from ghosts. */
        for ( cid = 0 ; cid < s->nr_ghost ; cid++ ) {
            epot += s->cells[ s->cid_ghost[cid] ].epot;
        }

        // **  get K1, calculate forces at current position **
        // set the integrator flag to set any persistent forces
        e->integrator_flags |= INTEGRATOR_UPDATE_PERSISTENTFORCE;
        if (engine_force( e ) < 0 ) {
            return error(engine_err);
        }
        e->integrator_flags &= ~INTEGRATOR_UPDATE_PERSISTENTFORCE;

#pragma omp parallel private(cid,c,pid,p,w,k,delta,c_dest,epot_local,ke)
        {
            step = omp_get_num_threads(); epot_local = 0.0;
            toofast = 0;
            for ( cid = omp_get_thread_num() ; cid < s->nr_real ; cid += step ) {
                c = &(s->cells[ s->cid_real[cid] ]);
                epot_local += c->epot;
                pid = 0;
                
                while ( pid < c->count ) {
                    p = &( c->parts[pid] );
                    if(engine::types[p->typeId].dynamics == PARTICLE_NEWTONIAN) {
                        p->vk[0] = p->force * p->imass;
                        p->xk[0] = p->velocity;
                    }
                    else {
                        p->xk[0] = p->force * p->imass;
                    }

                    // update position for k2
                    p->p0 = p->position;
                    p->v0 = p->velocity;
                    p->position = p->p0 + 0.5 * dt * p->xk[0];
                    CHECK_TOOFAST(p, h, h2);
                    pid += 1;
                }
            }
        }

        // ** get K2, calculate forces at x0 + 1/2 dt k1 **
        if (engine_force( e ) < 0 ) {
            return error(engine_err);
        }

#pragma omp parallel private(cid,c,pid,p,w,k,delta,c_dest,epot_local,ke)
        {
            step = omp_get_num_threads(); epot_local = 0.0;
            for ( cid = omp_get_thread_num() ; cid < s->nr_real ; cid += step ) {
                c = &(s->cells[ s->cid_real[cid] ]);
                epot_local += c->epot;
                pid = 0;
                while ( pid < c->count ) {
                    p = &( c->parts[pid] );

                    if(engine::types[p->typeId].dynamics == PARTICLE_NEWTONIAN) {
                        p->vk[1] = p->force * p->imass;
                        p->xk[1] = p->v0 + 0.5 * dt * p->vk[0];
                    }
                    else {
                        p->xk[1] = p->force * p->imass;
                    }

                    // setup pos for next k3
                    p->position = p->p0 + 0.5 * dt * p->xk[1];
                    CHECK_TOOFAST(p, h, h2);
                    pid += 1;
                }
            }
        }

        // ** get K3, calculate forces at x0 + 1/2 dt k2 **
        if (engine_force( e ) < 0 ) {
            return error(engine_err);
        }

#pragma omp parallel private(cid,c,pid,p,w,k,delta,c_dest,epot_local,ke)
        {
            step = omp_get_num_threads(); epot_local = 0.0;
            for ( cid = omp_get_thread_num() ; cid < s->nr_real ; cid += step ) {
                c = &(s->cells[ s->cid_real[cid] ]);
                epot_local += c->epot;
                pid = 0;
                while ( pid < c->count ) {
                    p = &( c->parts[pid] );

                    if(engine::types[p->typeId].dynamics == PARTICLE_NEWTONIAN) {
                        p->vk[2] = p->force * p->imass;
                        p->xk[2] = p->v0 + 0.5 * dt * p->vk[1];
                    }
                    else {
                        p->xk[2] = p->force * p->imass;
                    }

                    // setup pos for next k3
                    p->position = p->p0 + dt * p->xk[2];
                    CHECK_TOOFAST(p, h, h2);
                    pid += 1;
                }
            }
        }

        // ** get K4, calculate forces at x0 + dt k3, final position calculation **
        if (engine_force( e ) < 0 ) {
            return error(engine_err);
        }

#pragma omp parallel private(cid,c,pid,p,w,k,delta,c_dest,epot_local,ke)
        {
            step = omp_get_num_threads(); epot_local = 0.0;
            for ( cid = omp_get_thread_num() ; cid < s->nr_real ; cid += step ) {
                c = &(s->cells[ s->cid_real[cid] ]);
                epot_local += c->epot;
                pid = 0;
                while ( pid < c->count ) {
                    p = &( c->parts[pid] );
                    toofast = 0;

                    if(engine::types[p->typeId].dynamics == PARTICLE_NEWTONIAN) {
                        p->vk[3] = p->imass * p->force;
                        p->xk[3] = p->v0 + dt * p->vk[2];
                        p->velocity = p->v0 + (dt/6.) * (p->vk[0] + 2*p->vk[1] + 2 * p->vk[2] + p->vk[3]);
                    }
                    else {
                        p->xk[3] = p->imass * p->force;
                    }
                    
                    p->position = p->p0 + (dt/6.) * (p->xk[0] + 2*p->xk[1] + 2 * p->xk[2] + p->xk[3]);

                    for(int k = 0; k < 3; ++k) {
                        delta[k] = __builtin_isgreaterequal( p->x[k] , h[k] ) - __builtin_isless( p->x[k] , 0.0 );
                        toofast = toofast || (p->x[k] >= h2[k] || p->x[k] <= -h[k]);
                    }

                    if(toofast) {
                        return toofast_error(p);
                    }

                    /* do we have to move this particle? */
                    if ( ( delta[0] != 0 ) || ( delta[1] != 0 ) || ( delta[2] != 0 ) ) {
                        for ( k = 0 ; k < 3 ; k++ ) {
                            p->x[k] -= delta[k] * h[k];
                        }

                        c_dest = &( s->cells[ space_cellid( s ,
                                (c->loc[0] + delta[0] + s->cdim[0]) % s->cdim[0] ,
                                (c->loc[1] + delta[1] + s->cdim[1]) % s->cdim[1] ,
                                (c->loc[2] + delta[2] + s->cdim[2]) % s->cdim[2] ) ] );

                        pthread_mutex_lock(&c_dest->cell_mutex);
                        space_cell_add_incomming( c_dest , p );
                        pthread_mutex_unlock(&c_dest->cell_mutex);

                        s->celllist[ p->id ] = c_dest;

                        // remove a particle from a cell. if the part was the last in the
                        // cell, simply dec the count, otherwise, move the last part
                        // in the cell to the ejected part's prev loc.
                        c->count -= 1;
                        if ( pid < c->count ) {
                            c->parts[pid] = c->parts[c->count];
                            s->partlist[ c->parts[pid].id ] = &( c->parts[pid] );
                        }
                    }
                    else {
                        pid += 1;
                    }
                }
            }
#pragma omp atomic
            epot += epot_local;
        }

        /* Welcome the new particles in each cell. */
#pragma omp parallel for schedule(static)
        for ( cid = 0 ; cid < s->nr_marked ; cid++ ) {
            space_cell_welcome( &(s->cells[ s->cid_marked[cid] ]) , s->partlist );
        }

    } // endif  NOT if ((e->flags & engine_flag_verlet) || (e->flags & engine_flag_mpi))

    /* Store the accumulated potential energy. */
    s->epot_nonbond += epot;
    s->epot += epot;

    /* return quietly */
    return engine_err_ok;
}


