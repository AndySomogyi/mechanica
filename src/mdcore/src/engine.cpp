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


/* include some standard header files */
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <MxParticleEvent.h>
#include <MxCluster.hpp>

/* Include conditional headers. */
#include "mdcore_config.h"
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

/* include local headers */
#include "cycle.h"
#include "errs.h"
#include "fptype.h"
#include "lock.h"
#include "MxParticle.h"
#include "space_cell.h"
#include "task.h"
#include "queue.h"
#include "space.h"
#include "MxPotential.h"
#include "runner.h"
#include "bond.h"
#include "rigid.h"
#include "angle.h"
#include "dihedral.h"
#include "exclusion.h"
#include "reader.h"
#include "engine.h"
#include "engine_advance.h"
#include "MxForce.h"
#include <iostream>

#pragma clang diagnostic ignored "-Wwritable-strings"

/** ID of the last error. */
int engine_err = engine_err_ok;

/** TODO, clean up this design for types and static engine. */
/** What is the maximum nr of types? */
int engine::max_type = 128;
int engine::nr_types = 0;

/**
 * The particle types.
 *
 * Currently initialized in _MxParticle_init
 */
MxParticleData *engine::types = NULL;


/* the error macro. */
#define error(id)				( engine_err = errs_register( id , engine_err_msg[-(id)] , __LINE__ , __FUNCTION__ , __FILE__ ) )

/* list of error messages. */
const char *engine_err_msg[30] = {
		"Nothing bad happened.",
		"An unexpected NULL pointer was encountered.",
		"A call to malloc failed, probably due to insufficient memory.",
		"An error occured when calling a space function.",
		"A call to a pthread routine failed.",
		"An error occured when calling a runner function.",
		"One or more values were outside of the allowed range.",
		"An error occured while calling a cell function.",
		"The computational domain is too small for the requested operation.",
		"mdcore was not compiled with MPI.",
		"An error occured while calling an MPI function.",
		"An error occured when calling a bond function.",
		"An error occured when calling an angle function.",
		"An error occured when calling a reader function.",
		"An error occured while interpreting the PSF file.",
		"An error occured while interpreting the PDB file.",
		"An error occured while interpreting the CPF file.",
		"An error occured when calling a potential function.",
		"An error occured when calling an exclusion function.",
		"An error occured while computing the bonded sets.",
		"An error occured when calling a dihedral funtion.",
		"An error occured when calling a CUDA funtion.",
		"mdcore was not compiled with CUDA support.",
		"CUDA support is only available in single-precision.",
		"Max. number of parts per cell exceeded.",
		"An error occured when calling a queue funtion.",
		"An error occured when evaluating a rigid constraint.",
		"Cell cutoff size doesn't work with METIS",
		"METIS library undefined",
        "Particles moving too fast",
};


/**
 * @brief Re-shuffle the particles in the engine.
 *
 * @param e The #engine on which to run.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */

int engine_shuffle ( struct engine *e ) {

	int cid, k;
	struct space_cell *c;
	struct space *s = &e->s;

	/* Flush the ghost cells (to avoid overlapping particles) */
#pragma omp parallel for schedule(static), private(cid)
    for ( cid = 0 ; cid < s->nr_ghost ; cid++ ) {
		space_cell_flush( &(s->cells[s->cid_ghost[cid]]) , s->partlist , s->celllist );
    }

	/* Shuffle the domain. */
    if ( space_shuffle_local( s ) < 0 ) {
		return error(engine_err_space);
    }

#ifdef WITH_MPI
	/* Get the incomming particle from other procs if needed. */
	if ( e->particle_flags & engine_flag_mpi )
		if ( engine_exchange_incomming( e ) < 0 )
			return error(engine_err);
#endif

/* Welcome the new particles in each cell, unhook the old ones. */
#pragma omp parallel for schedule(static), private(cid,c,k)
	for ( cid = 0 ; cid < s->nr_marked ; cid++ ) {
		c = &(s->cells[s->cid_marked[cid]]);
		if ( !(c->flags & cell_flag_ghost) )
			space_cell_welcome( c , s->partlist );
		else {
			for ( k = 0 ; k < c->incomming_count ; k++ )
				e->s.partlist[ c->incomming[k].id ] = NULL;
			c->incomming_count = 0;
		}
	}

	/* return quietly */
	return engine_err_ok;

}


/**
 * @brief Set all the engine timers to 0.
 *
 * @param e The #engine.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */

int engine_timers_reset ( struct engine *e ) {

	int k;

	/* Check input nonsense. */
	if ( e == NULL )
		return error(engine_err_null);

	/* Run through the timers and set them to 0. */
	for ( k = 0 ; k < engine_timer_last ; k++ )
		e->timers[k] = 0;

	/* What, that's it? */
	return engine_err_ok;

}


/**
 * @brief Check if the Verlet-list needs to be updated.
 *
 * @param e The #engine.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */

int engine_verlet_update ( struct engine *e ) {

	int cid, pid, k;
	double dx, w, maxdx = 0.0, skin;
	struct space_cell *c;
	struct MxParticle *p;
	struct space *s = &e->s;
	ticks tic;
#ifdef HAVE_OPENMP
	int step;
	double lmaxdx;
#endif

	/* Do we really need to do this? */
	if ( !(e->flags & engine_flag_verlet) )
		return engine_err_ok;

	/* Get the skin width. */
	skin = fmin( s->h[0] , fmin( s->h[1] , s->h[2] ) ) - s->cutoff;

	/* Get the maximum particle movement. */
	if ( !s->verlet_rebuild ) {

#ifdef HAVE_OPENMP
#pragma omp parallel private(c,cid,pid,p,dx,k,w,step,lmaxdx)
		{
			lmaxdx = 0.0; step = omp_get_num_threads();
			for ( cid = omp_get_thread_num() ; cid < s->nr_real ; cid += step ) {
				c = &(s->cells[s->cid_real[cid]]);
				for ( pid = 0 ; pid < c->count ; pid++ ) {
					p = &(c->parts[pid]);
					for ( dx = 0.0 , k = 0 ; k < 3 ; k++ ) {
						w = p->x[k] - c->oldx[ 4*pid + k ];
						dx += w*w;
					}
					lmaxdx = fmax( dx , lmaxdx );
				}
			}
#pragma omp critical
			maxdx = fmax( lmaxdx , maxdx );
		}
#else
        for ( cid = 0 ; cid < s->nr_real ; cid++ ) {
            c = &(s->cells[s->cid_real[cid]]);
            for ( pid = 0 ; pid < c->count ; pid++ ) {
                p = &(c->parts[pid]);
                for ( dx = 0.0 , k = 0 ; k < 3 ; k++ ) {
                    w = p->x[k] - c->oldx[ 4*pid + k ];
                    dx += w*w;
                }
                maxdx = fmax( dx , maxdx );
            }
        }
#endif

#ifdef WITH_MPI
/* Collect the maximum displacement from other nodes. */
if ( ( e->particle_flags & engine_flag_mpi ) && ( e->nr_nodes > 1 ) ) {
	/* Do not use in-place as it is buggy when async is going on in the background. */
	if ( MPI_Allreduce( MPI_IN_PLACE , &maxdx , 1 , MPI_DOUBLE , MPI_MAX , e->comm ) != MPI_SUCCESS )
		return error(engine_err_mpi);
}
#endif

        /* Are we still in the green? */
        maxdx = sqrt(maxdx);
        s->verlet_rebuild = ( 2.0*maxdx > skin );

	}

	/* Do we have to rebuild the Verlet list? */
	if ( s->verlet_rebuild ) {

		/* printf("engine_verlet_update: re-building verlet lists next step...\n");
        printf("engine_verlet_update: maxdx=%e, skin=%e.\n",maxdx,skin); */

		/* Wait for any unterminated exchange. */
		tic = getticks();
#ifdef WITH_MPI
		if ( e->particle_flags & engine_flag_async )
			if ( engine_exchange_wait( e ) < 0 )
				return error(engine_err);
#endif
        tic = getticks() - tic;
        e->timers[engine_timer_exchange1] += tic;
        e->timers[engine_timer_verlet] -= tic;

        /* Move the particles to the respecitve cells. */
        if ( engine_shuffle( e ) < 0 )
            return error(engine_err);

        /* Store the current positions as a reference. */
        #pragma omp parallel for schedule(static), private(cid,c,pid,p,k)
        for ( cid = 0 ; cid < s->nr_real ; cid++ ) {
            c = &(s->cells[s->cid_real[cid]]);
            if ( c->oldx == NULL || c->oldx_size < c->count ) {
                free(c->oldx);
                c->oldx_size = c->size + 20;
                c->oldx = (FPTYPE *)malloc( sizeof(FPTYPE) * 4 * c->oldx_size );
            }
            for ( pid = 0 ; pid < c->count ; pid++ ) {
                p = &(c->parts[pid]);
                for ( k = 0 ; k < 3 ; k++ )
                    c->oldx[ 4*pid + k ] = p->x[k];
            }
        }

        /* Set the maximum displacement to zero. */
        s->maxdx = 0;

	}

	/* Otherwise, just store the maximum displacement. */
	else
		s->maxdx = maxdx;

	/* All done! */
	return engine_err_ok;

}



/**
 * @brief Clear all particles from this #engine.
 *
 * @param e The #engine to flush.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */

int engine_flush ( struct engine *e ) {

	/* check input. */
	if ( e == NULL )
		return error(engine_err_null);

	/* Clear the space. */
	if ( space_flush( &e->s ) < 0 )
		return error(engine_err_space);

	/* done for now. */
	return engine_err_ok;

}


/**
 * @brief Clear all particles from this #engine's ghost cells.
 *
 * @param e The #engine to flush.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */

int engine_flush_ghosts ( struct engine *e ) {

	/* check input. */
	if ( e == NULL )
		return error(engine_err_null);

	/* Clear the space. */
	if ( space_flush_ghosts( &e->s ) < 0 )
		return error(engine_err_space);

	/* done for now. */
	return engine_err_ok;

}


/** 
 * @brief Set the explicit electrostatic potential.
 *
 * @param e The #engine.
 * @param ep The electrostatic #potential.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 *
 * If @c ep is not @c NULL, the flag #engine_flag_explepot is set,
 * otherwise it is cleared.
 */

int engine_setexplepot ( struct engine *e , struct MxPotential *ep ) {

	/* check inputs. */
	if ( e == NULL )
		return error(engine_err_null);

	/* was a potential supplied? */
	if ( ep != NULL ) {

		/* set the flag. */
		e->flags |= engine_flag_explepot;

		/* set the potential. */
		e->ep = ep;

	}

	/* otherwise, just clear the flag. */
	else
		e->flags &= ~engine_flag_explepot;

	/* done for now. */
	return engine_err_ok;

}


/**
 * @brief Unload a set of particle data from the #engine.
 *
 * @param e The #engine.
 * @param x An @c N times 3 array of the particle positions.
 * @param v An @c N times 3 array of the particle velocities.
 * @param type A vector of length @c N of the particle type IDs.
 * @param pid A vector of length @c N of the particle IDs.
 * @param vid A vector of length @c N of the particle virtual IDs.
 * @param q A vector of length @c N of the individual particle charges.
 * @param flags A vector of length @c N of the particle flags.
 * @param epot A pointer to a #double in which to store the total potential energy.
 * @param N the maximum number of particles.
 *
 * @return The number of particles unloaded or < 0 on
 *      error (see #engine_err).
 *
 * The fields @c x, @c v, @c type, @c pid, @c vid, @c q, @c epot and/or @c flags may be NULL.
 */

int engine_unload ( struct engine *e , double *x , double *v , int *type , int *pid , int *vid , double *q , unsigned int *flags , double *epot , int N ) {

	struct MxParticle *p;
	struct space_cell *c;
	int j, k, cid, count = 0, *ind;
	double epot_acc = 0.0;

	/* check the inputs. */
	if ( e == NULL )
		return error(engine_err_null);

	/* Allocate and fill the indices. */
	if ( ( ind = (int *)alloca( sizeof(int) * (e->s.nr_cells + 1) ) ) == NULL )
		return error(engine_err_malloc);
	ind[0] = 0;
	for ( k = 0 ; k < e->s.nr_cells ; k++ )
		ind[k+1] = ind[k] + e->s.cells[k].count;
	if ( ind[e->s.nr_cells] > N )
		return error(engine_err_range);

	/* Loop over each cell. */
#pragma omp parallel for schedule(static), private(cid,count,c,k,p,j), reduction(+:epot_acc)
	for ( cid = 0 ; cid < e->s.nr_cells ; cid++ ) {

		/* Get a hold of the cell. */
		c = &( e->s.cells[cid] );
		count = ind[cid];

		/* Collect the potential energy if requested. */
		epot_acc += c->epot;

		/* Loop over the parts in this cell. */
		for ( k = 0 ; k < c->count ; k++ ) {

			/* Get a hold of the particle. */
			p = &( c->parts[k] );

			/* get this particle's data, where requested. */
			if ( x != NULL )
				for ( j = 0 ; j < 3 ; j++ )
					x[count*3+j] = c->origin[j] + p->x[j];
			if ( v != NULL)
				for ( j = 0 ; j < 3 ; j++ )
					v[count*3+j] = p->v[j];
			if ( type != NULL )
				type[count] = p->typeId;
			if ( pid != NULL )
				pid[count] = p->id;
			if ( vid != NULL )
				vid[count] = p->vid;
			if ( q != NULL )
				q[count] = p->q;
			if ( flags != NULL )
				flags[count] = p->flags;

			/* Step-up the counter. */
			count += 1;

		}

	}

	/* Write back the potential energy, if requested. */
	if ( epot != NULL )
		*epot += epot_acc;

	/* to the pub! */
	return ind[e->s.nr_cells];

}


/**
 * @brief Unload a set of particle data from the marked cells of an #engine
 *
 * @param e The #engine.
 * @param x An @c N times 3 array of the particle positions.
 * @param v An @c N times 3 array of the particle velocities.
 * @param type A vector of length @c N of the particle type IDs.
 * @param pid A vector of length @c N of the particle IDs.
 * @param vid A vector of length @c N of the particle virtual IDs.
 * @param q A vector of length @c N of the individual particle charges.
 * @param flags A vector of length @c N of the particle flags.
 * @param epot A pointer to a #double in which to store the total potential energy.
 * @param N the maximum number of particles.
 *
 * @return The number of particles unloaded or < 0 on
 *      error (see #engine_err).
 *
 * The fields @c x, @c v, @c type, @c pid, @c vid, @c q, @c epot and/or @c flags may be NULL.
 */

int engine_unload_marked ( struct engine *e , double *x , double *v , int *type , int *pid , int *vid , double *q , unsigned int *flags , double *epot , int N ) {

	struct MxParticle *p;
	struct space_cell *c;
	int j, k, cid, count = 0, *ind;
	double epot_acc = 0.0;

	/* check the inputs. */
	if ( e == NULL )
		return error(engine_err_null);

	/* Allocate and fill the indices. */
	if ( ( ind = (int *)alloca( sizeof(int) * (e->s.nr_cells + 1) ) ) == NULL )
		return error(engine_err_malloc);
	ind[0] = 0;
	for ( k = 0 ; k < e->s.nr_cells ; k++ )
		if ( e->s.cells[k].flags & cell_flag_marked )
			ind[k+1] = ind[k] + e->s.cells[k].count;
		else
			ind[k+1] = ind[k];
	if ( ind[e->s.nr_cells] > N )
		return error(engine_err_range);

	/* Loop over each cell. */
#pragma omp parallel for schedule(static), private(cid,count,c,k,p,j), reduction(+:epot_acc)
	for ( cid = 0 ; cid < e->s.nr_marked ; cid++ ) {

		/* Get a hold of the cell. */
		c = &( e->s.cells[e->s.cid_marked[cid]] );
		count = ind[e->s.cid_marked[cid]];

		/* Collect the potential energy if requested. */
		epot_acc += c->epot;

		/* Loop over the parts in this cell. */
		for ( k = 0 ; k < c->count ; k++ ) {

			/* Get a hold of the particle. */
			p = &( c->parts[k] );

			/* get this particle's data, where requested. */
			if ( x != NULL )
				for ( j = 0 ; j < 3 ; j++ )
					x[count*3+j] = c->origin[j] + p->x[j];
			if ( v != NULL)
				for ( j = 0 ; j < 3 ; j++ )
					v[count*3+j] = p->v[j];
			if ( type != NULL )
				type[count] = p->typeId;
			if ( pid != NULL )
				pid[count] = p->id;
			if ( vid != NULL )
				vid[count] = p->vid;
			if ( q != NULL )
				q[count] = p->q;
			if ( flags != NULL )
				flags[count] = p->flags;

			/* Step-up the counter. */
			count += 1;

		}

	}

	/* Write back the potential energy, if requested. */
	if ( epot != NULL )
		*epot += epot_acc;

	/* to the pub! */
	return ind[e->s.nr_cells];

}


/**
 * @brief Unload real particles that may have wandered into a ghost cell.
 *
 * @param e The #engine.
 * @param x An @c N times 3 array of the particle positions.
 * @param v An @c N times 3 array of the particle velocities.
 * @param type A vector of length @c N of the particle type IDs.
 * @param pid A vector of length @c N of the particle IDs.
 * @param vid A vector of length @c N of the particle virtual IDs.
 * @param q A vector of length @c N of the individual particle charges.
 * @param flags A vector of length @c N of the particle flags.
 * @param epot A pointer to a #double in which to store the total potential energy.
 * @param N the maximum number of particles.
 *
 * @return The number of particles unloaded or < 0 on
 *      error (see #engine_err).
 *
 * The fields @c x, @c v, @c type, @c vid, @c pid, @c q, @c epot and/or @c flags may be NULL.
 */

int engine_unload_strays ( struct engine *e , double *x , double *v , int *type , int *pid , int *vid , double *q , unsigned int *flags , double *epot , int N ) {

	struct MxParticle *p;
	struct space_cell *c;
	int j, k, cid, count = 0;
	double epot_acc = 0.0;

	/* check the inputs. */
	if ( e == NULL )
		return error(engine_err_null);

	/* Loop over each cell. */
	for ( cid = 0 ; cid < e->s.nr_real ; cid++ ) {

		/* Get a hold of the cell. */
		c = &( e->s.cells[e->s.cid_real[cid]] );

		/* Collect the potential energy if requested. */
		epot_acc += c->epot;

		/* Loop over the parts in this cell. */
		for ( k = c->count-1 ; k >= 0 && !(c->parts[k].flags & PARTICLE_GHOST) ; k-- ) {

			/* Get a hold of the particle. */
			p = &( c->parts[k] );
			if ( p->flags & PARTICLE_GHOST )
				continue;

			/* get this particle's data, where requested. */
			if ( x != NULL )
				for ( j = 0 ; j < 3 ; j++ )
					x[count*3+j] = c->origin[j] + p->x[j];
			if ( v != NULL)
				for ( j = 0 ; j < 3 ; j++ )
					v[count*3+j] = p->v[j];
			if ( type != NULL )
				type[count] = p->typeId;
			if ( pid != NULL )
				pid[count] = p->id;
			if ( vid != NULL )
				vid[count] = p->vid;
			if ( q != NULL )
				q[count] = p->q;
			if ( flags != NULL )
				flags[count] = p->flags;

			/* increase the counter. */
			count += 1;

		}

	}

	/* Write back the potential energy, if requested. */
	if ( epot != NULL )
		*epot += epot_acc;

	/* to the pub! */
	return count;

}


/**
 * @brief Load a set of particle data.
 *
 * @param e The #engine.
 * @param x An @c N times 3 array of the particle positions.
 * @param v An @c N times 3 array of the particle velocities.
 * @param type A vector of length @c N of the particle type IDs.
 * @param pid A vector of length @c N of the particle IDs.
 * @param vid A vector of length @c N of the particle virtual IDs.
 * @param q A vector of length @c N of the individual particle charges.
 * @param flags A vector of length @c N of the particle flags.
 * @param N the number of particles to load.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 *
 * If the parameters @c v, @c flags, @c vid or @c q are @c NULL, then
 * these values are set to zero.
 */

int engine_load ( struct engine *e , double *x , double *v , int *type , int *pid , int *vid , double *q , unsigned int *flags , int N ) {

    struct MxParticle p = {};
	int j, k;

	/* check the inputs. */
	if ( e == NULL || x == NULL || type == NULL )
		return error(engine_err_null);

	/* init the velocity and charge in case not specified. */
	p.v[0] = 0.0; p.v[1] = 0.0; p.v[2] = 0.0;
	p.f[0] = 0.0; p.f[1] = 0.0; p.f[2] = 0.0;
	p.q = 0.0;
	p.flags = PARTICLE_NONE;

	/* loop over the entries. */
	for ( j = 0 ; j < N ; j++ ) {

		/* set the particle data. */
		p.typeId = type[j];
		if ( pid != NULL )
			p.id = pid[j];
		else
			p.id = j;
		if ( vid != NULL )
			p.vid = vid[j];
		if ( flags != NULL )
			p.flags = flags[j];
		if ( v != NULL )
			for ( k = 0 ; k < 3 ; k++ )
				p.v[k] = v[j*3+k];
		if ( q != 0 )
			p.q = q[j];

		/* add the part to the space. */
		if ( engine_addpart( e , &p , &x[3*j], NULL ) < 0 )
			return error(engine_err_space);

	}

	/* to the pub! */
	return engine_err_ok;

}


/**
 * @brief Load a set of particle data as ghosts
 *
 * @param e The #engine.
 * @param x An @c N times 3 array of the particle positions.
 * @param v An @c N times 3 array of the particle velocities.
 * @param type A vector of length @c N of the particle type IDs.
 * @param pid A vector of length @c N of the particle IDs.
 * @param vid A vector of length @c N of the particle virtual IDs.
 * @param q A vector of length @c N of the individual particle charges.
 * @param flags A vector of length @c N of the particle flags.
 * @param N the number of particles to load.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 *
 * If the parameters @c v, @c flags, @c vid or @c q are @c NULL, then
 * these values are set to zero.
 */

int engine_load_ghosts ( struct engine *e , double *x , double *v , int *type , int *pid , int *vid , double *q , unsigned int *flags , int N ) {

    struct MxParticle p = {};
	struct space *s;
	int j, k;

	/* check the inputs. */
	if ( e == NULL || x == NULL || type == NULL )
		return error(engine_err_null);

	/* Get a handle on the space. */
	s = &(e->s);

	/* init the velocity and charge in case not specified. */
	p.v[0] = 0.0; p.v[1] = 0.0; p.v[2] = 0.0;
	p.f[0] = 0.0; p.f[1] = 0.0; p.f[2] = 0.0;
	p.q = 0.0;
	p.flags = PARTICLE_GHOST;

	/* loop over the entries. */
	for ( j = 0 ; j < N ; j++ ) {

		/* set the particle data. */
		p.typeId = type[j];
		if ( pid != NULL )
			p.id = pid[j];
		else
			p.id = j;
		if ( vid != NULL )
			p.vid = vid[j];
		if ( flags != NULL )
			p.flags = flags[j] | PARTICLE_GHOST;
		if ( v != NULL )
			for ( k = 0 ; k < 3 ; k++ )
				p.v[k] = v[j*3+k];
		if ( q != 0 )
			p.q = q[j];

		/* add the part to the space. */
		if ( engine_addpart( e , &p , &x[3*j], NULL ) < 0 )
			return error(engine_err_space);

	}

	/* to the pub! */
	return engine_err_ok;

}


/**
 * @brief Look for a given type by name.
 *
 * @param e The #engine.
 * @param name The type name.
 *
 * @return The type ID or < 0 on error (see #engine_err).
 */
int engine_gettype ( struct engine *e , char *name ) {

	int k;

	/* check for nonsense. */
	if ( e == NULL || name == NULL )
		return error(engine_err_null);

	/* Loop over the types... */
	for ( k = 0 ; k < e->nr_types ; k++ ) {

		/* Compare the name. */
		if ( strcmp( e->types[k].name , name ) == 0 )
			return k;

	}

	/* Otherwise, nothing found... */
	return engine_err_range;

}


/**
 * @brief Look for a given type by its second name.
 *
 * @param e The #engine.
 * @param name2 The type name2.
 *
 * @return The type ID or < 0 on error (see #engine_err).
 */

int engine_gettype2 ( struct engine *e , char *name2 ) {

	int k;

	/* check for nonsense. */
	if ( e == NULL || name2 == NULL )
		return error(engine_err_null);

	/* Loop over the types... */
	for ( k = 0 ; k < e->nr_types ; k++ ) {

		/* Compare the name. */
		if ( strcmp( e->types[k].name2 , name2 ) == 0 )
			return k;

	}

	/* Otherwise, nothing found... */
	return engine_err_range;

}


/**
 * @brief Add a type definition.
 *
 * @param e The #engine.
 * @param mass The particle type mass.
 * @param charge The particle type charge.
 * @param name Particle name, can be @c NULL.
 * @param name2 Particle second name, can be @c NULL.
 *
 * @return The type ID or < 0 on error (see #engine_err).
 *
 * The particle type ID must be an integer greater or equal to 0
 * and less than the value @c max_type specified in #engine_init.
 */
int engine_addtype ( struct engine *e , double mass , double charge ,
        const char *name , const char *name2 ) {
    
    /* check for nonsense. */
    if ( e == NULL )
        return error(engine_err_null);
    if ( e->nr_types >= e->max_type )
        return error(engine_err_range);
    
    MxParticleType *type = MxParticleType_ForEngine(e, mass, charge, name, name2);
    return type != NULL ? type->id : -1;
}

/**
 * @brief Add an interaction potential.
 *
 * @param e The #engine.
 * @param p The #potential to add to the #engine.
 * @param i ID of particle type for this interaction.
 * @param j ID of second particle type for this interaction.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 *
 * Adds the given potential for pairs of particles of type @c i and @c j,
 * where @c i and @c j may be the same type ID.
 */

int engine_addpot ( struct engine *e , struct MxPotential *p , int i , int j ) {

	/* check for nonsense. */
	if ( e == NULL )
		return error(engine_err_null);
	if ( i < 0 || i >= e->nr_types || j < 0 || j >= e->nr_types )
		return error(engine_err_range);
    
    MxPotential **pots = p->flags & POTENTIAL_BOUND ? e->p_bound : e->p;

	/* store the potential. */
	pots[ i * e->max_type + j ] = p;
    Py_INCREF(p);
    
    if ( i != j ) {
		pots[ j * e->max_type + i ] = p;
        Py_INCREF(p);
    }

	/* end on a good note. */
	return engine_err_ok;
}

CAPI_FUNC(int) engine_addforce1 ( struct engine *e , struct MxForce *p , int i ) {
    /* check for nonsense. */
    if ( e == NULL )
        return error(engine_err_null);
    if ( i < 0 || i >= e->nr_types  )
        return error(engine_err_range);
        
    /* store the force. */
    e->p_singlebody[i] = p;
    Py_INCREF(p);

    /* end on a good note. */
    return engine_err_ok;
}


/**
 * @brief Start the runners in the given #engine.
 *
 * @param e The #engine to start.
 * @param nr_runners The number of runners start.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 *
 * Allocates and starts the specified number of #runner. Also initializes
 * the Verlet lists.
 */

int engine_start ( struct engine *e , int nr_runners , int nr_queues ) {

	int cid, pid, k, i;
	struct space_cell *c;
	struct MxParticle *p;
	struct space *s = &e->s;

	/* Is MPI really needed? */
	if ( e->flags & engine_flag_mpi && e->nr_nodes == 1 )
		e->flags &= ~( engine_flag_mpi | engine_flag_async );

#ifdef WITH_MPI
	/* Set up async communication? */
	if ( e->particle_flags & engine_flag_async ) {

		/* Init the mutex and condition variable for the asynchronous communication. */
		if ( pthread_mutex_init( &e->xchg_mutex , NULL ) != 0 ||
				pthread_cond_init( &e->xchg_cond , NULL ) != 0 ||
				pthread_mutex_init( &e->xchg2_mutex , NULL ) != 0 ||
				pthread_cond_init( &e->xchg2_cond , NULL ) != 0 )
			return error(engine_err_pthread);

		/* Set the exchange flags. */
		e->xchg_started = 0;
		e->xchg_running = 0;
		e->xchg2_started = 0;
		e->xchg2_running = 0;

		/* Start a thread with the async exchange. */
		if ( pthread_create( &e->thread_exchg , NULL , (void *(*)(void *))engine_exchange_async_run , e ) != 0 )
			return error(engine_err_pthread);
		if ( pthread_create( &e->thread_exchg2 , NULL , (void *(*)(void *))engine_exchange_rigid_async_run , e ) != 0 )
			return error(engine_err_pthread);

	}
#endif

	/* Fill-in the Verlet lists if needed. */
	if ( e->flags & engine_flag_verlet ) {

		/* Shuffle the domain. */
		if ( engine_shuffle( e ) < 0 )
			return error(engine_err);

		/* Store the current positions as a reference. */
#pragma omp parallel for schedule(static), private(cid,c,pid,p,k)
		for ( cid = 0 ; cid < s->nr_real ; cid++ ) {
			c = &(s->cells[s->cid_real[cid]]);
			if ( c->oldx == NULL || c->oldx_size < c->count ) {
				free(c->oldx);
				c->oldx_size = c->size + 20;
				c->oldx = (FPTYPE *)malloc( sizeof(FPTYPE) * 4 * c->oldx_size );
			}
			for ( pid = 0 ; pid < c->count ; pid++ ) {
				p = &(c->parts[pid]);
				for ( k = 0 ; k < 3 ; k++ )
					c->oldx[ 4*pid + k ] = p->x[k];
			}
		}

		/* Re-set the Verlet rebuild flag. */
		s->verlet_rebuild = 1;

	}

	/* Is MPI really needed? */
	if ( e->flags & engine_flag_mpi && e->nr_nodes == 1 )
		e->flags &= ~engine_flag_mpi;

	/* Do we even need runners? */
	if ( e->flags & engine_flag_cuda ) {

		/* Set the number of runners. */
		e->nr_runners = nr_runners;

#if defined(HAVE_CUDA) && defined(WITH_CUDA)
		/* Load the potentials and pairs to the CUDA device. */
		if ( engine_cuda_load( e ) < 0 )
			return error(engine_err);
#else
		/* Was not compiled with CUDA support. */
		return error(engine_err_nocuda);
#endif

	}
	else {

		/* Allocate the queues */
		if ( ( e->queues = (struct queue *)malloc( sizeof(struct queue) * nr_queues )) == NULL )
			return error(engine_err_malloc);
		e->nr_queues = nr_queues;

		/* Initialize  and fill the queues. */
		for ( i = 0 ; i < e->nr_queues ; i++ )
			if ( queue_init( &e->queues[i] , 2*s->nr_tasks/e->nr_queues , s , s->tasks ) != queue_err_ok )
				return error(engine_err_queue);
		for ( i = 0 ; i < s->nr_tasks ; i++ )
			if ( queue_insert( &e->queues[ i % e->nr_queues ] , &s->tasks[i] ) < 0 )
				return error(engine_err_queue);

		/* (Allocate the runners */
				if ( ( e->runners = (struct runner *)malloc( sizeof(struct runner) * nr_runners )) == NULL )
					return error(engine_err_malloc);
				e->nr_runners = nr_runners;

				/* initialize the runners. */
				for ( i = 0 ; i < nr_runners ; i++ )
					if ( runner_init( &e->runners[ i ] , e , i ) < 0 )
						return error(engine_err_runner);

				/* wait for the runners to be in place */
				while (e->barrier_count != e->nr_runners)
					if (pthread_cond_wait(&e->done_cond,&e->barrier_mutex) != 0)
						return error(engine_err_pthread);

	}

	/* Set the number of runners. */
	e->nr_runners = nr_runners;

	/* all is well... */
	return engine_err_ok;
}

/**
 * @brief Compute the nonbonded interactions in the current step.
 * 
 * @param e The #engine on which to run.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 *
 * This routine advances the timestep counter by one, prepares the #space
 * for a timestep, releases the #runner's associated with the #engine
 * and waits for them to finnish.
 */

int engine_nonbond_eval ( struct engine *e ) {

	int k;

	/* Re-set the queues. */
	for ( k = 0 ; k < e->nr_queues ; k++ )
		e->queues[k].next = 0;

	/* open the door for the runners */
	e->barrier_count = -e->barrier_count;
	if ( e->nr_runners == 1 ) {
		if (pthread_cond_signal(&e->barrier_cond) != 0)
			return error(engine_err_pthread);
	}
	else {
		if (pthread_cond_broadcast(&e->barrier_cond) != 0)
			return error(engine_err_pthread);
	}

	/* wait for the runners to come home */
	while (e->barrier_count < e->nr_runners)
		if (pthread_cond_wait(&e->done_cond,&e->barrier_mutex) != 0)
			return error(engine_err_pthread);

	/* All in a days work. */
	return engine_err_ok;

}


/**
 * @brief Run the engine for a single time step.
 *
 * @param e The #engine on which to run.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 *
 * This routine advances the timestep counter by one, prepares the #space
 * for a timestep, releases the #runner's associated with the #engine
 * and waits for them to finnish.
 *
 * Once all the #runner's are done, the particle velocities and positions
 * are updated and the particles are re-sorted in the #space.
 */
int engine_step ( struct engine *e ) {

    ticks tic = getticks(), tic_step = tic;

	/* increase the time stepper */
	e->time += 1;

	engine_advance(e);


    e->timers[engine_timer_advance] += getticks() - tic;

    /* Shake the particle positions? */
    if ( e->nr_rigids > 0 ) {

#ifdef WITH_MPI
		/* If we have to do some communication first... */
		if ( e->particle_flags & engine_flag_mpi ) {

			/* Sort the constraints. */
			tic = getticks();
			if ( engine_rigid_sort( e ) != 0 )
				return error(engine_err);
			e->timers[engine_timer_rigid] += getticks() - tic;

			/* Start the clock. */
			tic = getticks();

			if ( e->particle_flags & engine_flag_async ) {
				if ( engine_exchange_rigid_async( e ) != 0 )
					return error(engine_err);
			}
			else {
				if ( engine_exchange_rigid( e ) != 0 )
					return error(engine_err);
			}

			/* Store the timing. */
			e->timers[engine_timer_exchange2] += getticks() - tic;

		}
#endif

		/* Resolve the constraints. */
		tic = getticks();
		if ( engine_rigid_eval( e ) != 0 )
			return error(engine_err);
		e->timers[engine_timer_rigid] += getticks() - tic;

	}

	/* Stop the clock. */
	e->timers[engine_timer_step] += getticks() - tic_step;

    // notify time listeners
    if(!SUCCEEDED(CMulticastTimeEvent_Invoke(e->on_time, e->time * e->dt))) {
        return error(engine_err);
    }

	/* return quietly */
	return engine_err_ok;
}

int engine_force(struct engine *e) {

    ticks tic = getticks();

    // clear the energy on the types
    // TODO: should go in prepare space for better performance
    engine_kinetic_energy(e);

    /* prepare the space, sets forces to zero */
    tic = getticks();
    if ( space_prepare( &e->s ) != space_err_ok )
        return error(engine_err_space);
    e->timers[engine_timer_prepare] += getticks() - tic;

    /* Make sure the verlet lists are up to date. */
    if ( e->flags & engine_flag_verlet ) {

        /* Start the clock. */
        tic = getticks();

        /* Check particle movement and update cells if necessary. */
        if ( engine_verlet_update( e ) < 0 ) {
            return error(engine_err);
        }

        /* Store the timing. */
        e->timers[engine_timer_verlet] += getticks() - tic;
    }
    

    /* Otherwise, if async MPI, move the particles accross the
       node boundaries. */
    else { // if ( e->flags & engine_flag_async ) {
        tic = getticks();
        if ( engine_shuffle( e ) < 0 ) {
            return error(engine_err_space);
        }
        e->timers[engine_timer_advance] += getticks() - tic;
    }
    

#ifdef WITH_MPI
    /* Re-distribute the particles to the processors. */
    if ( e->particle_flags & engine_flag_mpi ) {

        /* Start the clock. */
        tic = getticks();

        if ( e->particle_flags & engine_flag_async ) {
            if ( engine_exchange_async( e ) < 0 )
                return error(engine_err);
        }
        else {
            if ( engine_exchange( e ) < 0 )
                return error(engine_err);
        }

        /* Store the timing. */
        e->timers[engine_timer_exchange1] += getticks() - tic;

    }
#endif

    /* Compute the non-bonded interactions. */
    tic = getticks();

    if ( engine_nonbond_eval( e ) < 0 ) {
        return error(engine_err);
    }

    e->timers[engine_timer_nonbond] += getticks() - tic;

    /* Clear the verlet-rebuild flag if it was set. */
    if ( e->flags & engine_flag_verlet && e->s.verlet_rebuild )
        e->s.verlet_rebuild = 0;

    /* Do bonded interactions. */
    tic = getticks();
    if ( e->flags & engine_flag_sets ) {
        if ( engine_bonded_eval_sets( e ) < 0 )
            return error(engine_err);
    }
    else {
        if ( engine_bonded_eval( e ) < 0 )
            return error(engine_err);
    }
    e->timers[engine_timer_bonded] += getticks() - tic;


    return engine_err_ok;
}


/**
 * @brief Barrier routine to hold the @c runners back.
 *
 * @param e The #engine to wait on.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 *
 * After being initialized, and after every timestep, every #runner
 * calls this routine which blocks until all the runners have returned
 * and the #engine signals the next timestep.
 */

int engine_barrier ( struct engine *e ) {

	/* lock the barrier mutex */
	if (pthread_mutex_lock(&e->barrier_mutex) != 0)
		return error(engine_err_pthread);

	/* wait for the barrier to close */
	while (e->barrier_count < 0)
		if (pthread_cond_wait(&e->barrier_cond,&e->barrier_mutex) != 0)
			return error(engine_err_pthread);

	/* if i'm the last thread in, signal that the barrier is full */
	if (++e->barrier_count == e->nr_runners) {
		if (pthread_cond_signal(&e->done_cond) != 0)
			return error(engine_err_pthread);
	}

	/* wait for the barrier to re-open */
	while (e->barrier_count > 0)
		if (pthread_cond_wait(&e->barrier_cond,&e->barrier_mutex) != 0)
			return error(engine_err_pthread);

	/* if i'm the last thread out, signal to those waiting to get back in */
	if (++e->barrier_count == 0)
		if (pthread_cond_broadcast(&e->barrier_cond) != 0)
			return error(engine_err_pthread);

	/* free the barrier mutex */
	if (pthread_mutex_unlock(&e->barrier_mutex) != 0)
		return error(engine_err_pthread);

	/* all is well... */
	return engine_err_ok;

}


/**
 * @brief Initialize an #engine with the given data and MPI enabled.
 *
 * @param e The #engine to initialize.
 * @param origin An array of three doubles containing the cartesian origin
 *      of the space.
 * @param dim An array of three doubles containing the size of the space.
 * @param L The minimum cell edge length, should be at least @c cutoff.
 * @param cutoff The maximum interaction cutoff to use.
 * @param period A bitmask describing the periodicity of the domain
 *      (see #space_periodic_full).
 * @param max_type The maximum number of particle types that will be used
 *      by this engine.
 * @param flags Bit-mask containing the flags for this engine.
 * @param comm The MPI comm to use.
 * @param rank The ID of this node.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */

#ifdef WITH_MPI
int engine_init_mpi ( struct engine *e , const double *origin , const double *dim , double *L , double cutoff , unsigned int period , int max_type , unsigned int particle_flags , MPI_Comm comm , int rank ) {

	/* Init the engine. */
	if ( engine_init( e , origin , dim , L , cutoff , period , max_type , particle_flags | engine_flag_mpi ) < 0 )
		return error(engine_err);

	/* Store the MPI Comm and rank. */
	e->comm = comm;
	e->nodeID = rank;

	/* Bail. */
	return engine_err_ok;

}
#endif


/**
 * @brief Kill all runners and de-allocate the data of an engine.
 *
 * @param e the #engine to finalize.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */

int engine_finalize ( struct engine *e ) {

    int j, k;

    /* make sure the inputs are ok */
    if ( e == NULL )
        return error(engine_err_null);

    /* Shut down the runners, if they were started. */
    if ( e->runners != NULL ) {
        for ( k = 0 ; k < e->nr_runners ; k++ )
            if ( pthread_cancel( e->runners[k].thread ) != 0 )
                return error(engine_err_pthread);
        free( e->runners );
        free( e->queues );
    }

    /* Finalize the space. */
    // if ( space_finalize( &e->s ) < 0 )
    //     return error(engine_err_space);

    /* Free-up the types. */
    free( e->types );

    /* Free the potentials. */
    if ( e->p != NULL ) {
        for ( j = 0 ; j < e->nr_types ; j++ ) {
            for ( k = j ; k < e->nr_types ; k++ ) {
                if ( e->p[ j*e->max_type + k ] != NULL )
                    potential_clear( e->p[ j*e->max_type + k ] );
            }
        }

        for ( j = 0 ; j < e->nr_types ; j++ ) {
            for ( k = j ; k < e->nr_types ; k++ ) {
                if ( e->p[ j*e->max_type + k ] != NULL )
                    potential_clear( e->p_bound[ j*e->max_type + k ] );
            }
        }

        for ( k = 0 ; k < e->nr_dihedralpots ; k++ )
            potential_clear( e->p_dihedral[k] );
        free( e->p );
    }

    if ( e->p_dihedral != NULL )
        free( e->p_dihedral );

    /* Free the communicators, if needed. */
    if ( e->flags & engine_flag_mpi ) {
        for ( k = 0 ; k < e->nr_nodes ; k++ ) {
            free( e->send[k].cellid );
            free( e->recv[k].cellid );
        }
        free( e->send );
        free( e->recv );
    }

    /* Free the bonded interactions. */
    free( e->bonds );
    free( e->angles );
    free( e->dihedrals );
    free( e->exclusions );
    free( e->rigids );
    free( e->part2rigid );

    /* If we have bonded sets, kill them. */
    for ( k = 0 ; k < e->nr_sets ; k++ ) {
        free( e->sets[k].bonds );
        free( e->sets[k].angles );
        free( e->sets[k].dihedrals );
        free( e->sets[k].exclusions );
        free( e->sets[k].confl );
    }

    /* Clear all the counts and what not. */
    bzero( e , sizeof( struct engine ) );

    /* Happy and I know it... */
    return engine_err_ok;

}


int engine_init ( struct engine *e , const double *origin , const double *dim , double *L ,
        double cutoff , unsigned int period , int max_type , unsigned int flags ) {

    int cid;

    /* make sure the inputs are ok */
    if ( e == NULL || origin == NULL || dim == NULL || L == NULL )
        return error(engine_err_null);

    /* default Boltzmann constant to 1 */
    e->K = 1.0;

    e->integrator_flags = 0;

    /* Check for bad flags. */
#ifdef FPTYPE_DOUBLE
    if ( e->particle_flags & engine_flag_cuda )
        return error(engine_err_cudasp);
#endif

    /* init the space with the given parameters */
    if ( space_init( &(e->s) , origin , dim , L , cutoff , period ) < 0 )
        return error(engine_err_space);

    /* Set some flag implications. */
    if ( flags & engine_flag_verlet_pseudo )
        flags |= engine_flag_verlet_pairwise;
    if ( flags & engine_flag_verlet_pairwise )
        flags |= engine_flag_verlet;
    if ( flags & engine_flag_cuda )
        flags |= engine_flag_nullpart;

    /* Set the flags. */
    e->flags = flags;

    /* By default there is only one node. */
    e->nr_nodes = 1;

    /* Init the timers. */
    if ( engine_timers_reset( e ) < 0 )
        return error(engine_err);

    /* Init the runners to 0. */
    e->runners = NULL;
    e->nr_runners = 0;

    /* Start with no queues. */
    e->queues = NULL;
    e->nr_queues = 0;

    /* Init the bonds array. */
    e->bonds_size = 100;
    if ( ( e->bonds = (struct MxBond *)malloc( sizeof( struct MxBond ) * e->bonds_size ) ) == NULL )
        return error(engine_err_malloc);
    e->nr_bonds = 0;
    e->nr_active_bonds = 0;

    /* Init the exclusions array. */
    e->exclusions_size = 100;
    if ( ( e->exclusions = (struct exclusion *)malloc( sizeof( struct exclusion ) * e->exclusions_size ) ) == NULL )
        return error(engine_err_malloc);
    e->nr_exclusions = 0;

    /* Init the rigids array. */
    e->rigids_size = 100;
    if ( ( e->rigids = (struct rigid *)malloc( sizeof( struct rigid ) * e->rigids_size ) ) == NULL )
        return error(engine_err_malloc);
    e->nr_rigids = 0;
    e->tol_rigid = 1e-6;
    e->nr_constr = 0;
    e->part2rigid = NULL;

    /* Init the angles array. */
    e->angles_size = 100;
    if ( ( e->angles = (struct MxAngle *)malloc( sizeof( struct MxAngle ) * e->angles_size ) ) == NULL )
        return error(engine_err_malloc);
    e->nr_angles = 0;

    /* Init the dihedrals array.		 */
    e->dihedrals_size = 100;
    if ( ( e->dihedrals = (struct dihedral *)malloc( sizeof( struct dihedral ) * e->dihedrals_size ) ) == NULL )
        return error(engine_err_malloc);
    e->nr_dihedrals = 0;

    
    /* Init the sets. */
    e->sets = NULL;
    e->nr_sets = 0;

    /* allocate the interaction matrices */
    if ( ( e->p = (struct MxPotential **)malloc( sizeof(MxPotential*) * e->max_type * e->max_type ) ) == NULL )
        return error(engine_err_malloc);
    
    /* allocate the flux interaction matrices */
    if ( ( e->fluxes = ( MxFluxes **)malloc( sizeof(MxFluxes*) * e->max_type * e->max_type ) ) == NULL )
        return error(engine_err_malloc);

    if ( ( e->p_bound = (struct MxPotential **)malloc( sizeof(MxPotential*) * e->max_type * e->max_type ) ) == NULL )
            return error(engine_err_malloc);

    bzero( e->p , sizeof(struct MxPotential *) * e->max_type * e->max_type );
    
    bzero( e->fluxes , sizeof(struct MxFluxes *) * e->max_type * e->max_type );

    bzero( e->p_bound , sizeof(struct MxPotential *) * e->max_type * e->max_type );

    e->dihedralpots_size = 100;
    if ( (e->p_dihedral = (struct MxPotential **)malloc( sizeof(struct MxPotential *) * e->dihedralpots_size )) == NULL)
        return error(engine_err_malloc);
    bzero( e->p_dihedral , sizeof(struct MxPotential *) * e->dihedralpots_size );
    e->nr_dihedralpots = 0;

    // init singlebody forces
    if ( ( e->p_singlebody = (MxForce **)malloc( sizeof(MxForce *) * e->max_type ) ) == NULL )
            return error(engine_err_malloc);
    bzero(e->p_singlebody, sizeof(struct MxForce *) * e->max_type );

    /* Make sortlists? */
    if ( flags & engine_flag_verlet_pseudo ) {
        for ( cid = 0 ; cid < e->s.nr_cells ; cid++ )
            if ( e->s.cells[cid].flags & cell_flag_marked )
                if ( ( e->s.cells[cid].sortlist = (unsigned int *)malloc( sizeof(unsigned int) * 13 * e->s.cells[cid].size ) ) == NULL )
                    return error(engine_err_malloc);
    }

    /* init the barrier variables */
    e->barrier_count = 0;
    if ( pthread_mutex_init( &e->barrier_mutex , NULL ) != 0 ||
            pthread_cond_init( &e->barrier_cond , NULL ) != 0 ||
            pthread_cond_init( &e->done_cond , NULL ) != 0)
        return error(engine_err_pthread);

    /* init the barrier */
    if (pthread_mutex_lock(&e->barrier_mutex) != 0)
        return error(engine_err_pthread);
    e->barrier_count = 0;

    /* Init the comm arrays. */
    e->send = NULL;
    e->recv = NULL;
    
    e->on_time = CMulticastTimeEvent_New();

    e->integrator = EngineIntegrator::FORWARD_EULER;

    e->flags |= engine_flag_initialized;
    
    e->particle_max_dist_fraction = 0.05;

    /* all is well... */
    return engine_err_ok;

}




void engine_dump() {
    for(int cid = 0; cid < _Engine.s.nr_cells; ++cid) {
        space_cell *cell = &_Engine.s.cells[cid];
        for(int pid = 0; pid < cell->count; ++pid) {
            MxParticle *p = &cell->parts[pid];

            std::cout << "i: " << pid << ", pid: " << p->id <<
                    ", {" << p->x[0] << ", " << p->x[1] << ", " << p->x[2] << "}"
                    ", {" << p->v[0] << ", " << p->v[1] << ", " << p->v[2] << "}\n";

        }
    }
}

double engine_kinetic_energy(struct engine *e)
{
    // clear the ke in the types,
    for(int i = 0; i < engine::nr_types; ++i) {
        engine::types[i].kinetic_energy = 0;
    }
    
    for(int cid = 0; cid < _Engine.s.nr_cells; ++cid) {
        space_cell *cell = &_Engine.s.cells[cid];
        for(int pid = 0; pid < cell->count; ++pid) {
            MxParticle *p = &cell->parts[pid];
            engine::types[p->typeId].kinetic_energy += engine::types[p->typeId].mass *
                    (p->v[0] * p->v[0] + p->v[1] * p->v[1] + p->v[2] * p->v[2]);
        }
    }
    
    for(int i = 1; i < engine::nr_types; ++i) {
        engine::types[0].kinetic_energy += engine::types[i].kinetic_energy;
        engine::types[i].kinetic_energy = engine::types[i].kinetic_energy / (2. * engine::types[i].parts.nr_parts);
    }
    
    // TODO: super lame hack to work around bug with
    // not setting root particle count. FIX THIS. 
    engine::types[0].kinetic_energy /= (2. * engine::types[0].parts.nr_parts);
    return engine::types[0].kinetic_energy;
}

double engine_temperature(struct engine *e)
{
    return 0;
}

int engine_singlebody_set(struct engine *e, struct MxForce *f, int type_id)
{
    if (type_id >= e->max_type) {
        return error(engine_err_range);
    }

    if(e->p_singlebody[type_id]) {
        Py_DECREF(e->p_singlebody[type_id]);
        e->p_singlebody[type_id] = NULL;
    }

    if(f) {
        e->p_singlebody[type_id] = f;
        Py_INCREF(f);
    }

    /* all is well... */
    return engine_err_ok;
}

int engine_addpart(struct engine *e, struct MxParticle *p, double *x,
        struct MxParticle **result)
{
    if(p->typeId < 0 || p->typeId >= e->nr_types) {
        return error(engine_err_range);
    }

    if(space_addpart (&(e->s), p, x, result ) != 0) {
        return error(engine_err_space);
    }

    e->types[p->typeId].addpart(p->id);
    
    return engine_err_ok;
}

int engine_addcuboid(struct engine *e, struct MxCuboid *p, struct MxCuboid **result)
{

    if(space_addcuboid(&(e->s), p, result ) != 0) {
        return error(engine_err_space);
    }
    
    return engine_err_ok;
}

CAPI_FUNC(struct MxParticleType*) engine_type(int id)
{
    if(id >= 0 && id < engine::nr_types) {
        return &engine::types[id];
    }
    return NULL;
}

int engine_next_partid(struct engine *e)
{
    // TODO: not the most effecient algorithm...
    space *s = &e->s;
    int i;
    
    for(i = 0; i < s->nr_parts; ++i) {
        if(s->partlist[i] == NULL) {
            return i;
        }
    }
    return i;
}

CAPI_FUNC(HRESULT) engine_del_particle(struct engine *e, int pid)
{
    std::cout << "time: " << e->time * e->dt << ", deleting particle id: " << pid << std::endl;
    
    if(pid < 0 || pid >= e->s.size_parts) {
        return c_error(E_FAIL, "pid out of range");
    }
    
    MxParticle *part = e->s.partlist[pid];
    
    if(part == NULL) {
        return c_error(E_FAIL, "particle already null");
    }
    
    MxParticleType *type = &e->types[part->typeId];
    
    HRESULT hr = type->del_part(pid);
    if(!SUCCEEDED(hr)) {
        return hr;
    }
    
    std::vector<int32_t> bonds = MxBond_IdsForParticle(pid);
    
    for(int i = 0; i < bonds.size(); ++i) {
        MxBond_Destroy(&_Engine.bonds[bonds[i]]);
    }
    
    return space_del_particle(&e->s, pid);
}


Magnum::Vector3 engine_center() {
    Magnum::Vector3 dim = {
        (float)_Engine.s.dim[0],
        (float)_Engine.s.dim[1],
        (float)_Engine.s.dim[2]
    };
    return dim / 2.;
}

