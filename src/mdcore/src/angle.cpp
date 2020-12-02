/*******************************************************************************
 * This file is part of mdcore.
 * Coypright (c) 2010 Pedro Gonnet (pedro.gonnet@durham.ac.uk)
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 ******************************************************************************/

#include <angle.h>

/* Include configuration header */
#include "mdcore_config.h"

/* Include some standard header files */
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <limits.h>

/* Include some conditional headers. */
#include "mdcore_config.h"
#ifdef __SSE__
    #include <xmmintrin.h>
#endif
#ifdef WITH_MPI
    #include <mpi.h>
#endif

/* Include local headers */
#include "cycle.h"
#include "errs.h"
#include "fptype.h"
#include "lock.h"
#include <MxParticle.h>
#include <MxPotential.h>
#include "potential_eval.h"
#include <space_cell.h>
#include "space.h"
#include "engine.h"
#include "MxPy.h"

#include <iostream>



/* Global variables. */
/** The ID of the last error. */
int angle_err = angle_err_ok;
unsigned int angle_rcount = 0;

/* the error macro. */
#define error(id)				( angle_err = errs_register( id , angle_err_msg[-(id)] , __LINE__ , __FUNCTION__ , __FILE__ ) )

/* list of error messages. */
const char *angle_err_msg[2] = {
	"Nothing bad happened.",
    "An unexpected NULL pointer was encountered."
	};


static int angle_init(MxAngle*, PyObject *, PyObject *);

static MxAngle *angle_alloc(PyTypeObject *type, Py_ssize_t);

    

/**
 * @brief Evaluate a list of angleed interactions
 *
 * @param b Pointer to an array of #angle.
 * @param N Nr of angles in @c b.
 * @param e Pointer to the #engine in which these angles are evaluated.
 * @param epot_out Pointer to a double in which to aggregate the potential energy.
 * 
 * @return #angle_err_ok or <0 on error (see #angle_err)
 */
 
int angle_eval ( struct MxAngle *a , int N , struct engine *e , double *epot_out ) {

    int aid, pid, pjd, pkd, k, *loci, *locj, *lock, shift;
    double h[3], epot = 0.0;
    struct space *s;
    struct MxParticle *pi, *pj, *pk, **partlist;
    struct space_cell **celllist;
    struct MxPotential *pot;
    Magnum::Vector3 xi, xj, xk, dxi, dxk;
    FPTYPE ctheta, wi, wk;
    Magnum::Vector3 rji, rjk;
    FPTYPE inji, injk, dprod;
#if defined(VECTORIZE)
    struct MxPotential *potq[VEC_SIZE];
    int icount = 0;
    FPTYPE *effi[VEC_SIZE], *effj[VEC_SIZE], *effk[VEC_SIZE];
    FPTYPE cthetaq[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE ee[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE eff[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE diq[VEC_SIZE*3], dkq[VEC_SIZE*3];
#else
    FPTYPE ee, eff;
#endif
    
    /* Check inputs. */
    if ( a == NULL || e == NULL )
        return error(angle_err_null);
        
    /* Get local copies of some variables. */
    s = &e->s;
    partlist = s->partlist;
    celllist = s->celllist;
    for ( k = 0 ; k < 3 ; k++ )
        h[k] = s->h[k];
        
    /* Loop over the angles. */
    for ( aid = 0 ; aid < N ; aid++ ) {
    
        /* Get the particles involved. */
        pid = a[aid].i; pjd = a[aid].j; pkd = a[aid].k;
        if ( ( pi = partlist[ pid] ) == NULL )
            continue;
        if ( ( pj = partlist[ pjd ] ) == NULL )
            continue;
        if ( ( pk = partlist[ pkd ] ) == NULL )
            continue;
        
        /* Skip if all three are ghosts. */
        if ( ( pi->flags & PARTICLE_GHOST ) && ( pj->flags & PARTICLE_GHOST ) && ( pk->flags & PARTICLE_GHOST ) )
            continue;
            
        /* Get the potential. */
        if ( ( pot = a[aid].potential) == NULL )
            continue;
    
        /* get the particle positions relative to pj's cell. */
        loci = celllist[ pid ]->loc;
        locj = celllist[ pjd ]->loc;
        lock = celllist[ pkd ]->loc;
        for ( k = 0 ; k < 3 ; k++ ) {
            xj[k] = pj->x[k];
            shift = loci[k] - locj[k];
            if ( shift > 1 )
                shift = -1;
            else if ( shift < -1 )
                shift = 1;
            xi[k] = pi->x[k] + shift*h[k];
            shift = lock[k] - locj[k];
            if ( shift > 1 )
                shift = -1;
            else if ( shift < -1 )
                shift = 1;
            xk[k] = pk->x[k] + shift*h[k];
        }
            
        /* Get the angle rays. */
        for ( k = 0 ; k < 3 ; k++ ) {
            rji[k] = xi[k] - xj[k];
            rjk[k] = xk[k] - xj[k];
        }
            
        /* Compute some quantities we will re-use. */
        dprod = rji[0]*rjk[0] + rji[1]*rjk[1] + rji[2]*rjk[2];
        inji = FPTYPE_ONE / FPTYPE_SQRT( rji[0]*rji[0] + rji[1]*rji[1] + rji[2]*rji[2] );
        injk = FPTYPE_ONE / FPTYPE_SQRT( rjk[0]*rjk[0] + rjk[1]*rjk[1] + rjk[2]*rjk[2] );
        
        /* Compute the cosine. */
        ctheta = FPTYPE_FMAX( -FPTYPE_ONE , FPTYPE_FMIN( FPTYPE_ONE , dprod * inji * injk ) );
        
        // Set the derivatives.
        // particles could be perpenducular, then plan is undefined, so
        // choose a random orientation plane
        if(ctheta == 0 || ctheta == -1) {
            std::uniform_real_distribution<float> dist{-1, 1};
            // make a random vector
            Magnum::Vector3 x{dist(CRandom), dist(CRandom), dist(CRandom)};
            
            // vector between outer particles
            Magnum::Vector3 vik = xi - xk;
            
            // make it orthogonal to rji
            x = x - Magnum::Math::dot(x, vik) * vik;
            
            // normalize it.
            dxi = dxk = x.normalized();
        } else {
            for ( k = 0 ; k < 3 ; k++ ) {
                dxi[k] = ( rjk[k]*injk - ctheta * rji[k]*inji ) * inji;
                dxk[k] = ( rji[k]*inji - ctheta * rjk[k]*injk ) * injk;
            }
        }
        
        /* printf( "angle_eval: cos of angle %i (%s-%s-%s) is %e.\n" , aid ,
            e->types[pi->type].name , e->types[pj->type].name , e->types[pk->type].name , ctheta ); */
        /* printf( "angle_eval: ids are ( %i , %i , %i ).\n" , pi->id , pj->id , pk->id );
        if ( e->s.celllist[pid] != e->s.celllist[pjd] )
            printf( "angle_eval: pi and pj are in different cells!\n" );
        if ( e->s.celllist[pkd] != e->s.celllist[pjd] )
            printf( "angle_eval: pk and pj are in different cells!\n" );
        printf( "angle_eval: xi-xj is [ %e , %e , %e ], ||xi-xj||=%e.\n" ,
            xi[0]-xj[0] , xi[1]-xj[1] , xi[2]-xj[2] , sqrt( (xi[0]-xj[0])*(xi[0]-xj[0]) + (xi[1]-xj[1])*(xi[1]-xj[1]) + (xi[2]-xj[2])*(xi[2]-xj[2]) ) );
        printf( "angle_eval: xk-xj is [ %e , %e , %e ], ||xk-xj||=%e.\n" ,
            xk[0]-xj[0] , xk[1]-xj[1] , xk[2]-xj[2] , sqrt( (xk[0]-xj[0])*(xk[0]-xj[0]) + (xk[1]-xj[1])*(xk[1]-xj[1]) + (xk[2]-xj[2])*(xk[2]-xj[2]) ) ); */
        /* printf( "angle_eval: dxi is [ %e , %e , %e ], ||dxi||=%e.\n" ,
            dxi[0] , dxi[1] , dxi[2] , sqrt( dxi[0]*dxi[0] + dxi[1]*dxi[1] + dxi[2]*dxi[2] ) );
        printf( "angle_eval: dxk is [ %e , %e , %e ], ||dxk||=%e.\n" ,
            dxk[0] , dxk[1] , dxk[2] , sqrt( dxk[0]*dxk[0] + dxk[1]*dxk[1] + dxk[2]*dxk[2] ) ); */
        if ( ctheta < pot->a || ctheta > pot->b ) {
            printf( "angle_eval[%i]: angle %i (%s-%s-%s) out of range [%e,%e], ctheta=%e.\n" ,
                e->nodeID , aid , e->types[pi->typeId].name , e->types[pj->typeId].name , e->types[pk->typeId].name , pot->a , pot->b , ctheta );
            ctheta = FPTYPE_FMAX( pot->a , FPTYPE_FMIN( pot->b , ctheta ) );
        }

        #ifdef VECTORIZE
            /* add this angle to the interaction queue. */
            cthetaq[icount] = ctheta;
            diq[icount*3] = dxi[0];
            diq[icount*3+1] = dxi[1];
            diq[icount*3+2] = dxi[2];
            dkq[icount*3] = dxk[0];
            dkq[icount*3+1] = dxk[1];
            dkq[icount*3+2] = dxk[2];
            effi[icount] = pi->f;
            effj[icount] = pj->f;
            effk[icount] = pk->f;
            potq[icount] = pot;
            icount += 1;

            /* evaluate the interactions if the queue is full. */
            if ( icount == VEC_SIZE ) {

                #if defined(FPTYPE_SINGLE)
                    #if VEC_SIZE==8
                    potential_eval_vec_8single_r( potq , cthetaq , ee , eff );
                    #else
                    potential_eval_vec_4single_r( potq , cthetaq , ee , eff );
                    #endif
                #elif defined(FPTYPE_DOUBLE)
                    #if VEC_SIZE==4
                    potential_eval_vec_4double_r( potq , cthetaq , ee , eff );
                    #else
                    potential_eval_vec_2double_r( potq , cthetaq , ee , eff );
                    #endif
                #endif

                /* update the forces and the energy */
                for ( l = 0 ; l < VEC_SIZE ; l++ ) {
                    epot += ee[l];
                    for ( k = 0 ; k < 3 ; k++ ) {
                        effi[l][k] -= ( wi = eff[l] * diq[3*l+k] );
                        effk[l][k] -= ( wk = eff[l] * dkq[3*l+k] );
                        effj[l][k] += wi + wk;
                        }
                    }

                /* re-set the counter. */
                icount = 0;

                }
        #else
            /* evaluate the angle */
            #ifdef EXPLICIT_POTENTIALS
                potential_eval_expl( pot , ctheta , &ee , &eff );
            #else
                potential_eval_r( pot , ctheta , &ee , &eff );
            #endif
            
            /* update the forces */
            for ( k = 0 ; k < 3 ; k++ ) {
                pi->f[k] -= ( wi = eff * dxi[k] );
                pk->f[k] -= ( wk = eff * dxk[k] );
                pj->f[k] += wi + wk;
            }

            /* tabulate the energy */
            epot += ee;
        #endif
        
        } /* loop over angles. */
        
        
    #if defined(VECTORIZE)
        /* are there any leftovers? */
        if ( icount > 0 ) {

            /* copy the first potential to the last entries */
            for ( k = icount ; k < VEC_SIZE ; k++ ) {
                potq[k] = potq[0];
                cthetaq[k] = cthetaq[0];
                }

            /* evaluate the potentials */
            #if defined(FPTYPE_SINGLE)
                #if VEC_SIZE==8
                potential_eval_vec_8single_r( potq , cthetaq , ee , eff );
                #else
                potential_eval_vec_4single_r( potq , cthetaq , ee , eff );
                #endif
            #elif defined(FPTYPE_DOUBLE)
                #if VEC_SIZE==4
                potential_eval_vec_4double_r( potq , cthetaq , ee , eff );
                #else
                potential_eval_vec_2double_r( potq , cthetaq , ee , eff );
                #endif
            #endif

            /* for each entry, update the forces and energy */
            for ( l = 0 ; l < icount ; l++ ) {
                epot += ee[l];
                for ( k = 0 ; k < 3 ; k++ ) {
                    effi[l][k] -= ( wi = eff[l] * diq[3*l+k] );
                    effk[l][k] -= ( wk = eff[l] * dkq[3*l+k] );
                    effj[l][k] += wi + wk;
                    }
                }

            }
    #endif
    
    /* Store the potential energy. */
    *epot_out += epot;
    
    /* We're done here. */
    return angle_err_ok;
    
    }


/**
 * @brief Evaluate a list of angleed interactions
 *
 * @param b Pointer to an array of #angle.
 * @param N Nr of angles in @c b.
 * @param e Pointer to the #engine in which these angles are evaluated.
 * @param epot_out Pointer to a double in which to aggregate the potential energy.
 *
 * This function differs from #angle_eval in that the forces are added to
 * the array @c f instead of directly in the particle data.
 * 
 * @return #angle_err_ok or <0 on error (see #angle_err)
 */
 
int angle_evalf ( struct MxAngle *a , int N , struct engine *e , FPTYPE *f , double *epot_out ) {

    int aid, pid, pjd, pkd, k, *loci, *locj, *lock, shift;
    double h[3], epot = 0.0;
    struct space *s;
    struct MxParticle *pi, *pj, *pk, **partlist;
    struct space_cell **celllist;
    struct MxPotential *pot;
    FPTYPE xi[3], xj[3], xk[3], dxi[3] , dxk[3], ctheta, wi, wk;
    FPTYPE t1, t10, t11, t12, t13, t21, t22, t23, t24, t25, t26, t27, t3,
        t5, t6, t7, t8, t9, t4, t14, t2;

#if defined(VECTORIZE)
    struct MxPotential *potq[VEC_SIZE];
    int icount = 0, l;
    FPTYPE *effi[VEC_SIZE], *effj[VEC_SIZE], *effk[VEC_SIZE];
    FPTYPE cthetaq[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE ee[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE eff[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE diq[VEC_SIZE*3], dkq[VEC_SIZE*3];
#else
    FPTYPE ee, eff;
#endif
    
    /* Check inputs. */
    if ( a == NULL || e == NULL )
        return error(angle_err_null);
        
    /* Get local copies of some variables. */
    s = &e->s;
    partlist = s->partlist;
    celllist = s->celllist;
    for ( k = 0 ; k < 3 ; k++ )
        h[k] = s->h[k];
        
    /* Loop over the angles. */
    for ( aid = 0 ; aid < N ; aid++ ) {
    
        /* Get the particles involved. */
        pid = a[aid].i; pjd = a[aid].j; pkd = a[aid].k;
        if ( ( pi = partlist[ pid] ) == NULL )
            continue;
        if ( ( pj = partlist[ pjd ] ) == NULL )
            continue;
        if ( ( pk = partlist[ pkd ] ) == NULL )
            continue;
        
        /* Skip if all three are ghosts. */
        if ( ( pi->flags & PARTICLE_GHOST ) && ( pj->flags & PARTICLE_GHOST ) && ( pk->flags & PARTICLE_GHOST ) )
            continue;
            
        /* Get the potential. */
        if ( ( pot = a[aid].potential ) == NULL )
            continue;
    
        /* get the particle positions relative to pj's cell. */
        loci = celllist[ pid ]->loc;
        locj = celllist[ pjd ]->loc;
        lock = celllist[ pkd ]->loc;
        for ( k = 0 ; k < 3 ; k++ ) {
            xj[k] = pj->x[k];
            shift = loci[k] - locj[k];
            if ( shift > 1 )
                shift = -1;
            else if ( shift < -1 )
                shift = 1;
            xi[k] = pi->x[k] + h[k]*shift;
            shift = lock[k] - locj[k];
            if ( shift > 1 )
                shift = -1;
            else if ( shift < -1 )
                shift = 1;
            xk[k] = pk->x[k] + h[k]*shift;
            }
            
        /* This is Maple-generated code, see "angles.maple" for details. */
        t2 = xj[2]*xj[2];
        t4 = xj[1]*xj[1];
        t14 = xj[0]*xj[0];
        t21 = t2+t4+t14;
        t24 = -FPTYPE_TWO*xj[2];
        t25 = -FPTYPE_TWO*xj[1];
        t26 = -FPTYPE_TWO*xj[0];
        t6 = (t24+xi[2])*xi[2]+(t25+xi[1])*xi[1]+(t26+xi[0])*xi[0]+t21;
        t3 = FPTYPE_ONE/sqrt(t6);
        t10 = xk[0]-xj[0];
        t11 = xi[2]-xj[2];
        t12 = xi[1]-xj[1];
        t13 = xi[0]-xj[0];
        t8 = xk[2]-xj[2];
        t9 = xk[1]-xj[1];
        t7 = t13*t10+t12*t9+t11*t8;
        t27 = t3*t7;
        t5 = (t24+xk[2])*xk[2]+(t25+xk[1])*xk[1]+(t26+xk[0])*xk[0]+t21;
        t1 = FPTYPE_ONE/sqrt(t5);
        t23 = t1/t5*t7;
        t22 = FPTYPE_ONE/t6*t27;
        dxi[0] = (t10*t3-t13*t22)*t1;
        dxi[1] = (t9*t3-t12*t22)*t1;
        dxi[2] = (t8*t3-t11*t22)*t1;
        dxk[0] = (t13*t1-t10*t23)*t3;
        dxk[1] = (t12*t1-t9*t23)*t3;
        dxk[2] = (t11*t1-t8*t23)*t3;
        ctheta = FPTYPE_FMAX( -FPTYPE_ONE , FPTYPE_FMIN( FPTYPE_ONE , t1*t27 ) );
        
        /* printf( "angle_eval: angle %i is %e rad.\n" , aid , ctheta ); */
        if ( ctheta < pot->a || ctheta > pot->b ) {
            printf( "angle_evalf: angle %i (%s-%s-%s) out of range [%e,%e], ctheta=%e.\n" ,
                aid , e->types[pi->typeId].name , e->types[pj->typeId].name , e->types[pk->typeId].name , pot->a , pot->b , ctheta );
            ctheta = fmax( pot->a , fmin( pot->b , ctheta ) );
            }

        #ifdef VECTORIZE
            /* add this angle to the interaction queue. */
            cthetaq[icount] = ctheta;
            diq[icount*3] = dxi[0];
            diq[icount*3+1] = dxi[1];
            diq[icount*3+2] = dxi[2];
            dkq[icount*3] = dxk[0];
            dkq[icount*3+1] = dxk[1];
            dkq[icount*3+2] = dxk[2];
            effi[icount] = &f[ 4*pid ];
            effj[icount] = &f[ 4*pjd ];
            effk[icount] = &f[ 4*pkd ];
            potq[icount] = pot;
            icount += 1;

            /* evaluate the interactions if the queue is full. */
            if ( icount == VEC_SIZE ) {

                #if defined(FPTYPE_SINGLE)
                    #if VEC_SIZE==8
                    potential_eval_vec_8single_r( potq , cthetaq , ee , eff );
                    #else
                    potential_eval_vec_4single_r( potq , cthetaq , ee , eff );
                    #endif
                #elif defined(FPTYPE_DOUBLE)
                    #if VEC_SIZE==4
                    potential_eval_vec_4double_r( potq , cthetaq , ee , eff );
                    #else
                    potential_eval_vec_2double_r( potq , cthetaq , ee , eff );
                    #endif
                #endif

                /* update the forces and the energy */
                for ( l = 0 ; l < VEC_SIZE ; l++ ) {
                    epot += ee[l];
                    for ( k = 0 ; k < 3 ; k++ ) {
                        effi[l][k] -= ( wi = eff[l] * diq[3*l+k] );
                        effk[l][k] -= ( wk = eff[l] * dkq[3*l+k] );
                        effj[l][k] += wi + wk;
                        }
                    }

                /* re-set the counter. */
                icount = 0;

                }
        #else
            /* evaluate the angle */
            #ifdef EXPLICIT_POTENTIALS
                potential_eval_expl( pot , ctheta , &ee , &eff );
            #else
                potential_eval_r( pot , ctheta , &ee , &eff );
            #endif
            
            /* update the forces */
            for ( k = 0 ; k < 3 ; k++ ) {
                f[4*pid+k] -= ( wi = eff * dxi[k] );
                f[4*pkd+k] -= ( wk = eff * dxk[k] );
                f[4*pjd+k] += wi + wk;
                }

            /* tabulate the energy */
            epot += ee;
        #endif
        
        } /* loop over angles. */
        
    #if defined(VECTORIZE)
        /* are there any leftovers? */
        if ( icount > 0 ) {

            /* copy the first potential to the last entries */
            for ( k = icount ; k < VEC_SIZE ; k++ ) {
                potq[k] = potq[0];
                cthetaq[k] = cthetaq[0];
                }

            /* evaluate the potentials */
            #if defined(FPTYPE_SINGLE)
                #if VEC_SIZE==8
                potential_eval_vec_8single_r( potq , cthetaq , ee , eff );
                #else
                potential_eval_vec_4single_r( potq , cthetaq , ee , eff );
                #endif
            #elif defined(FPTYPE_DOUBLE)
                #if VEC_SIZE==4
                potential_eval_vec_4double_r( potq , cthetaq , ee , eff );
                #else
                potential_eval_vec_2double_r( potq , cthetaq , ee , eff );
                #endif
            #endif

            /* for each entry, update the forces and energy */
            for ( l = 0 ; l < icount ; l++ ) {
                epot += ee[l];
                for ( k = 0 ; k < 3 ; k++ ) {
                    effi[l][k] -= ( wi = eff[l] * diq[3*l+k] );
                    effk[l][k] -= ( wk = eff[l] * dkq[3*l+k] );
                    effj[l][k] += wi + wk;
                    }
                }

            }
    #endif
    
    /* Store the potential energy. */
    *epot_out += epot;
    
    /* We're done here. */
    return angle_err_ok;
    
}

MxAngle* MxAngle_NewFromIds(int i, int j, int k, int pid)
{
}

MxAngle* MxAngle_NewFromIdsAndPotential(int i, int j, int k,
        struct MxPotential *pot)
{
}



PyTypeObject MxAngle_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "Angle",
    .tp_basicsize = sizeof(MxAngle),
    .tp_itemsize =       0,
    .tp_dealloc =        0,
    .tp_print =          0,
    .tp_getattr =        0,
    .tp_setattr =        0,
    .tp_as_async =       0,
    .tp_repr =           0,
    .tp_as_number =      0,
    .tp_as_sequence =    0,
    .tp_as_mapping =     0,
    .tp_hash =           0,
    .tp_call =           0,
    .tp_str =            0,
    .tp_getattro =       0,
    .tp_setattro =       0,
    .tp_as_buffer =      0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "Custom objects",
    .tp_traverse =       0,
    .tp_clear =          0,
    .tp_richcompare =    0,
    .tp_weaklistoffset = 0,
    .tp_iter =           0,
    .tp_iternext =       0,
    .tp_methods =        0,
    .tp_members =        0,
    .tp_getset =         0,
    .tp_base =           0,
    .tp_dict =           0,
    .tp_descr_get =      0,
    .tp_descr_set =      0,
    .tp_dictoffset =     0,
    .tp_init =           (initproc)angle_init,
    .tp_alloc =          (allocfunc)angle_alloc,
    .tp_new =            PyType_GenericNew,
    .tp_free =           0,
    .tp_is_gc =          0,
    .tp_bases =          0,
    .tp_mro =            0,
    .tp_cache =          0,
    .tp_subclasses =     0,
    .tp_weaklist =       0,
    .tp_del =            0,
    .tp_version_tag =    0,
    .tp_finalize =       0,
};

HRESULT _MxAngle_init(PyObject *module)
{
    if (PyType_Ready((PyTypeObject*)&MxAngle_Type) < 0) {
        std::cout << "could not initialize MxAngle_Type " << std::endl;
        return E_FAIL;
    }


    Py_INCREF(&MxAngle_Type);
    if (PyModule_AddObject(module, "Angle", (PyObject *)&MxAngle_Type) < 0) {
        Py_DECREF(&MxAngle_Type);
        return E_FAIL;
    }

    return S_OK;
}

int angle_init(MxAngle *self, PyObject *args, PyObject *kwargs) {
    
    std::cout << MX_FUNCTION << std::endl;
    
    try {
        PyObject *pot  = arg<PyObject*>("potential", 0, args, kwargs);
        PyObject *p1  = arg<PyObject*>("p1", 1, args, kwargs);
        PyObject *p2  = arg<PyObject*>("p2", 2, args, kwargs);
        PyObject *p3  = arg<PyObject*>("p3", 3, args, kwargs);
        
        
        if(PyObject_IsInstance(pot, (PyObject*)&MxPotential_Type) <= 0) {
            PyErr_SetString(PyExc_TypeError, "potential is not a instance of Potential");
            return -1;
        }
        
        if(MxParticle_Check(p1) <= 0) {
            PyErr_SetString(PyExc_TypeError, "p1 is not a instance of Particle");
            return -1;
        }
        
        if(MxParticle_Check(p2) <= 0) {
            PyErr_SetString(PyExc_TypeError, "p2 is not a instance Particle");
            return -1;
        }
        
        if(MxParticle_Check(p3) <= 0) {
            PyErr_SetString(PyExc_TypeError, "p3 is not a instance Particle");
            return -1;
        }
        
        self->potential = (MxPotential*)pot;
        self->i = ((MxParticleHandle*)p1)->id;
        self->j = ((MxParticleHandle*)p2)->id;
        self->k = ((MxParticleHandle*)p3)->id;
        
        Py_XINCREF(pot);
    }
    catch (const std::exception &e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        return -1;
    }
    catch(pybind11::error_already_set &e){
        e.restore();
        return -1;
    }
    return 0;
}

MxAngle *angle_alloc(PyTypeObject *type, Py_ssize_t) {
    MxAngle *result = NULL;
    uint32_t err = engine_angle_alloc(&_Engine, type, &result);
    return result;
}
