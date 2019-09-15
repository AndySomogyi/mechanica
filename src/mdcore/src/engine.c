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

/* Include conditional headers. */
#include "config.h"
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
#include <particle.h>
#include <space_cell.h>
#include "task.h"
#include "queue.h"
#include "space.h"
#include "potential.h"
#include "runner.h"
#include "bond.h"
#include "rigid.h"
#include "angle.h"
#include "dihedral.h"
#include "exclusion.h"
#include "reader.h"
#include "engine.h"


/** ID of the last error. */
int engine_err = engine_err_ok;


/* the error macro. */
#define error(id)				( engine_err = errs_register( id , engine_err_msg[-(id)] , __LINE__ , __FUNCTION__ , __FILE__ ) )

/* list of error messages. */
char *engine_err_msg[29] = {
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
	for ( cid = 0 ; cid < s->nr_ghost ; cid++ )
		space_cell_flush( &(s->cells[s->cid_ghost[cid]]) , s->partlist , s->celllist );

	/* Shuffle the domain. */
	if ( space_shuffle_local( s ) < 0 )
		return error(engine_err_space);


#ifdef WITH_MPI
	/* Get the incomming particle from other procs if needed. */
	if ( e->flags & engine_flag_mpi )
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
	struct particle *p;
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
if ( ( e->flags & engine_flag_mpi ) && ( e->nr_nodes > 1 ) ) {
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
		if ( e->flags & engine_flag_async )
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
 * @brief Set-up the engine for distributed-memory parallel operation.
 *
 * @param e The #engine to set-up.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 *
 * This function assumes that #engine_split_bisect or some similar
 * function has already been called and that #nodeID, #nr_nodes as
 * well as the #cell @c nodeIDs have been set.
 */
int engine_split ( struct engine *e ) {

	int i, k, cid, cjd;
	struct space_cell *ci, *cj, *ct;
	struct space *s = &(e->s);

	/* Check for nonsense inputs. */
	if ( e == NULL )
		return error(engine_err_null);

	/* Start by allocating and initializing the send/recv lists. */
	if ( ( e->send = (struct engine_comm *)malloc( sizeof(struct engine_comm) * e->nr_nodes ) ) == NULL ||
			( e->recv = (struct engine_comm *)malloc( sizeof(struct engine_comm) * e->nr_nodes ) ) == NULL )
		return error(engine_err_malloc);
	for ( k = 0 ; k < e->nr_nodes ; k++ ) {
		if ( ( e->send[k].cellid = (int *)malloc( sizeof(int) * 100 ) ) == NULL )
			return error(engine_err_malloc);
		e->send[k].size = 100;
		e->send[k].count = 0;
		if ( ( e->recv[k].cellid = (int *)malloc( sizeof(int) * 100 ) ) == NULL )
			return error(engine_err_malloc);
		e->recv[k].size = 100;
		e->recv[k].count = 0;
	}

	/* Un-mark all cells. */
	for ( cid = 0 ; cid < s->nr_cells ; cid++ )
		s->cells[cid].flags &= ~cell_flag_marked;

	/* Loop over each cell pair... */
	for ( i = 0 ; i < s->nr_tasks ; i++ ) {

		/* Is this task a pair? */
		if ( s->tasks[i].type != task_type_pair )
			continue;

		/* Get the cells in this pair. */
		cid = s->tasks[i].i;
		cjd = s->tasks[i].j;
		ci = &( s->cells[ cid ] );
		cj = &( s->cells[ cjd ] );

		/* If it is a ghost-ghost pair, skip it. */
		if ( (ci->flags & cell_flag_ghost) && (cj->flags & cell_flag_ghost) )
			continue;

		/* Mark the cells. */
		ci->flags |= cell_flag_marked;
		cj->flags |= cell_flag_marked;

		/* Make cj the ghost cell and bail if both are real. */
		if ( ci->flags & cell_flag_ghost ) {
			ct = ci; ci = cj; cj = ct;
			k = cid; cid = cjd; cjd = k;
		}
		else if ( !( cj->flags & cell_flag_ghost ) )
			continue;

		/* Store the communication between cid and cjd. */
		/* Store the send, if not already there... */
		for ( k = 0 ; k < e->send[cj->nodeID].count && e->send[cj->nodeID].cellid[k] != cid ; k++ );
		if ( k == e->send[cj->nodeID].count ) {
			if ( e->send[cj->nodeID].count == e->send[cj->nodeID].size ) {
				e->send[cj->nodeID].size += 100;
				if ( ( e->send[cj->nodeID].cellid = (int *)realloc( e->send[cj->nodeID].cellid , sizeof(int) * e->send[cj->nodeID].size ) ) == NULL )
					return error(engine_err_malloc);
			}
			e->send[cj->nodeID].cellid[ e->send[cj->nodeID].count++ ] = cid;
		}
		/* Store the recv, if not already there... */
		for ( k = 0 ; k < e->recv[cj->nodeID].count && e->recv[cj->nodeID].cellid[k] != cjd ; k++ );
		if ( k == e->recv[cj->nodeID].count ) {
			if ( e->recv[cj->nodeID].count == e->recv[cj->nodeID].size ) {
				e->recv[cj->nodeID].size += 100;
				if ( ( e->recv[cj->nodeID].cellid = (int *)realloc( e->recv[cj->nodeID].cellid , sizeof(int) * e->recv[cj->nodeID].size ) ) == NULL )
					return error(engine_err_malloc);
			}
			e->recv[cj->nodeID].cellid[ e->recv[cj->nodeID].count++ ] = cjd;
		}

	}

	/* Nuke all ghost-ghost tasks. */
	i = 0;
	while ( i < s->nr_tasks ) {

		/* Pair? */
		if ( s->tasks[i].type == task_type_pair ) {

			/* Get the cells in this pair. */
			ci = &( s->cells[ s->tasks[i].i ] );
			cj = &( s->cells[ s->tasks[i].j ] );

			/* If it is a ghost-ghost pair, skip it. */
			if ( (ci->flags & cell_flag_ghost) && (cj->flags & cell_flag_ghost) )
				s->tasks[i] = s->tasks[ --(s->nr_tasks) ];
			else
				i += 1;

		}

		/* Self? */
		else if ( s->tasks[i].type == task_type_self ) {

			/* Get the cells in this pair. */
			ci = &( s->cells[ s->tasks[i].i ] );

			/* If it is a ghost-ghost pair, skip it. */
			if ( ci->flags & cell_flag_ghost )
				s->tasks[i] = s->tasks[ --(s->nr_tasks) ];
			else
				i += 1;

		}

		/* Sort? */
		else if ( s->tasks[i].type == task_type_sort ) {

			/* Get the cells in this pair. */
			ci = &( s->cells[ s->tasks[i].i ] );

			/* If it is a ghost-ghost pair, skip it. */
			if ( !(ci->flags & cell_flag_marked) )
				s->tasks[i] = s->tasks[ --(s->nr_tasks) ];
			else
				i += 1;

		}

	}

	/* Clear all task dependencies and re-link each sort task with its cell. */
	for ( i = 0 ; i < s->nr_tasks ; i++ ) {
		s->tasks[i].nr_unlock = 0;
		if ( s->tasks[i].type == task_type_sort ) {
			s->cells[ s->tasks[i].i ].sort = &s->tasks[i];
			s->tasks[i].flags = 0;
		}
	}

	/* Run through the tasks and make each pair depend on the sorts.
       Also set the flags for each sort. */
	for ( k = 0 ; k < s->nr_tasks ; k++ )
		if ( s->tasks[k].type == task_type_pair ) {
			if ( task_addunlock( s->cells[ s->tasks[k].i ].sort , &s->tasks[k] ) != 0 ||
					task_addunlock( s->cells[ s->tasks[k].j ].sort , &s->tasks[k] ) != 0 )
				return error(space_err_task);
			s->cells[ s->tasks[k].i ].sort->flags |= 1 << s->tasks[k].flags;
			s->cells[ s->tasks[k].j ].sort->flags |= 1 << s->tasks[k].flags;
		}


	/* Empty unmarked cells. */
	for ( k = 0 ; k < s->nr_cells ; k++ )
		if ( !( s->cells[k].flags & cell_flag_marked ) )
			space_cell_flush( &s->cells[k] , s->partlist , s->celllist );

	/* Set ghost markings on particles. */
	for ( cid = 0 ; cid < s->nr_cells ; cid++ )
		if ( s->cells[cid].flags & cell_flag_ghost )
			for ( k = 0 ; k < s->cells[cid].count ; k++ )
				s->cells[cid].parts[k].flags |= PARTICLE_FLAG_GHOST;

	/* Fill the cid lists with marked, local and ghost cells. */
	s->nr_real = 0; s->nr_ghost = 0; s->nr_marked = 0;
	for ( cid = 0 ; cid < s->nr_cells ; cid++ )
		if ( s->cells[cid].flags & cell_flag_marked ) {
			s->cid_marked[ s->nr_marked++ ] = cid;
			if ( s->cells[cid].flags & cell_flag_ghost ) {
				s->cells[cid].id = -s->nr_cells;
				s->cid_ghost[ s->nr_ghost++ ] = cid;
			}
			else {
				s->cells[cid].id = s->nr_real;
				s->cid_real[ s->nr_real++ ] = cid;
			}
		}

	/* Done deal. */
	return engine_err_ok;

}

#ifdef WITH_METIS
/**
 * @brief Split the computation domain over a number of nodes using METIS graph partitioning.
 *
 *@param e The #engine to split up.
 *@param N The number of computational nodes.
 *@param flags Flag telling whether to split the space for MPI or for GPUs.
 *
 *@return #engine_err_ok or < 0 on error (see #engine_err).
 */
int engine_split_METIS ( struct engine *e, int N, int flags){

	//printf("Using METIS algorithm to split the space\n");
	int currentIndex, i,j,shiftDim,neighbor;
	idx_t vw; //Temporary vertex Weight store
	idx_t ew; //Temporary edge Weight store


	//Do single GPU version ie. N = 1
	if(N==1)
	{
		if( flags == engine_split_MPI )
		{
			for(i = 0; i < e->s.nr_cells; i++ )
			{
				e->s.cells[i].nodeID = 0;

			}
		}else if ( flags == engine_split_GPU )
		{
			for( i = 0 ; i < e->s.nr_cells ; i++ )
			{
				e->s.cells[i].GPUID = 0;
			}


		}
		return engine_err_ok;
	}

	//Values to adjust weighting of edges dependent on spatial share.
	//These values are taken from random simulations of vector distances as described in GONNET 2007.
	float FACE = 0.50004f;
	float EDGE = 0.16176f;
	float CORNER = 0.036213f;

	int nr_pairs = 0;
	for( i = 0; i < e->s.nr_tasks; i++ )
		if(e->s.tasks[i].type == task_type_pair)
			nr_pairs++;
	nr_pairs *= 2;
	/* Check inputs. */
	if ( e == NULL )
		return error(engine_err_null);

	/* Check cell size >= cutoff distance*/
	if( e->s.h[0] < e->s.cutoff || e->s.h[1] < e->s.cutoff || e->s.h[2] < e->s.cutoff)
		return error(engine_err_cutoff);

	//Need to include #include <metis.h>
	/*Allocate memory required for METIS*/
	/*Number of cells = number of nodes. */
	idx_t *xadj = (idx_t*) malloc((e->s.nr_cells+1) * sizeof(idx_t));
	if(xadj == NULL) return error(engine_err_malloc);
	/*Number of edges = number of cellpairs */
	idx_t *adjncy = (idx_t*) malloc (nr_pairs *2 * sizeof(idx_t));
	if(adjncy == NULL) return error(engine_err_malloc);
	/*Vertex Weights */
	idx_t *vwgt = (idx_t*) malloc(e->s.nr_cells * sizeof(idx_t));
	if(vwgt == NULL) return error(engine_err_malloc);
	/*Edge Weights */
	idx_t *adjwgt = (idx_t*) malloc(nr_pairs * 2 * sizeof(idx_t));
	if(adjwgt == NULL) return error(engine_err_malloc);
	/*Number vertices */
	idx_t *nvtxs = (idx_t*) malloc(sizeof(idx_t));
	if(nvtxs==NULL)return 1;
	*nvtxs = e->s.nr_cells;
	//Number constraints
	idx_t *ncon = (idx_t*) malloc(sizeof(idx_t));
	if(ncon==NULL)return 1;
	*ncon = 1;
	//Number partitions
	idx_t *nparts = (idx_t*) malloc(sizeof(idx_t));
	if(nparts==NULL)return 1;
	*nparts = N;

	/*results*/
	idx_t *objval = (idx_t*) malloc(sizeof(idx_t));
	if(objval==NULL)return 1;
	idx_t *particle = (idx_t*) malloc(e->s.nr_cells * (sizeof(idx_t)));
	if(particle==NULL)return 1;

	//Loop over cell pairs and add to array if valid. Needs to be double loop.
	currentIndex=0;
	for(j=0; j<e->s.nr_cells; j++)
	{
		vw=0;
		ew=0;
		xadj[j]=currentIndex;
		for(i=0; i<e->s.nr_tasks; i++)
		{
			if(e->s.tasks[i].type == task_type_sort)
				continue;
			shiftDim=0;
			//If the pair involves cell j.
			if(e->s.tasks[i].i==j || e->s.tasks[i].j==j)
			{


				//If type = self this is a self interaction. All particles interact (n^2-n interactions)
				//Increment vertex weight by e->s.cell[j].count ^ 2 - e->s.cell[j].count
				if(e->s.tasks[i].type == task_type_self)
				{
					vw+= e->s.cells[j].count*e->s.cells[j].count - e->s.cells[j].count;
				}else{
					//Find neighbor cell index.
					if(e->s.tasks[i].i==j)
						neighbor = e->s.tasks[i].j;
					else
						neighbor = e->s.tasks[i].i;

					//Check how many shift==0.
					if(e->s.cells[e->s.tasks[i].i].loc[0] == e->s.cells[e->s.tasks[i].j].loc[0])
						shiftDim++;
					if(e->s.cells[e->s.tasks[i].i].loc[1] == e->s.cells[e->s.tasks[i].j].loc[1])
						shiftDim++;
					if(e->s.cells[e->s.tasks[i].i].loc[2] == e->s.cells[e->s.tasks[i].j].loc[2])
						shiftDim++;

					//If shiftDim = 2 this is a face interaction. Add edge from j to i
					//Edge has weight e->s.cell[j].count * e->s.cell[neighbor].count * FACE	(Estimated number of distance calculations)
					if(shiftDim==2)
					{
						ew = e->s.cells[j].count * e->s.cells[neighbor].count * FACE;
						//Vertex Weights are the sum of edge weights + self interaction
						vw+= ew;
						//Add an edge and the edge weight to the graph
						adjncy[currentIndex] = neighbor;
						adjwgt[currentIndex] = ew;
						currentIndex++;
					}

					//If shiftDim = 1 this is an edge interaction. Add edge from j to i
					//Edge has weight e->s.cell[j].count * e->s.cell[neightbor.count * EDGE (Estimated number of distance calculations)
					if(shiftDim==1)
					{
						ew = e->s.cells[j].count * e->s.cells[neighbor].count * EDGE;
						//Vertex Weights are the sum of edge weights + self interaction
						vw+=ew;
						//Add an edge and the edge weight to the graph
						adjncy[currentIndex] = neighbor;
						adjwgt[currentIndex] = ew;
						currentIndex++;
					}


					//If shiftDim = 0 this is an corner interaction. Add edge from j to i
					//Edge has weight e->s.cell[j].count * e->s.cell[neightbor.count * CORNER (Estimated number of distance calculations)
					if(shiftDim==0)
					{
						ew = e->s.cells[j].count * e->s.cells[neighbor].count * CORNER;
						//Vertex Weights are the sum of edge weights + self interaction
						vw+=ew;
						//Add an edge and the edge weight to the graph
						adjncy[currentIndex] = neighbor;
						adjwgt[currentIndex] = ew;
						currentIndex++;
					}
				}


			}
		}
		//We now know vertex weight.
		vwgt[j] = vw;
	}

	//Need to add the final thing METIS needs to xadj
	xadj[e->s.nr_cells]=currentIndex;

	//Setup METIS options
	idx_t *options = (idx_t*) malloc(METIS_NOPTIONS * sizeof(idx_t));
	if(options==NULL)return 1;
	METIS_SetDefaultOptions(options);
	options[METIS_OPTION_PTYPE]=METIS_PTYPE_KWAY;
	options[METIS_OPTION_OBJTYPE]=METIS_OBJTYPE_CUT;
	options[METIS_OPTION_CTYPE] = METIS_CTYPE_SHEM;
	options[METIS_OPTION_IPTYPE]=METIS_IPTYPE_GROW;
	options[METIS_OPTION_NCUTS] = 1;
	options[METIS_OPTION_NSEPS] = 1;
	options[METIS_OPTION_NUMBERING] = 0; //C-style numbering :)
	options[METIS_OPTION_NITER] = 10;
	options[METIS_OPTION_SEED] = 1;
	options[METIS_OPTION_CONTIG] = 1;
	options[METIS_OPTION_UFACTOR] = 30;
	//	options[METIS_OPTION_DBGLVL] = METIS_DBG_INFO;

	//Run METIS to partition the graph
	METIS_PartGraphKway( nvtxs , ncon , xadj , adjncy , vwgt , NULL , adjwgt , nparts , NULL , NULL , options , objval , particle );
	if( flags == engine_split_MPI )
	{
		for(i = 0; i < e->s.nr_cells; i++ )
		{
			e->s.cells[i].nodeID = particle[i];
			//Not my cell? Mark as ghost.
			if(particle[i] != e->nodeID)
				e->s.cells[i].flags |= cell_flag_ghost;
			e->nr_nodes = N;

		}
	}else if ( flags == engine_split_GPU )
	{
		int part1 = 0;
		int part2 = 0;
		for( i = 0 ; i < e->s.nr_cells ; i++ )
		{
			e->s.cells[i].GPUID = particle[i];
			if(e->s.cells[i].GPUID == 0)
				part1++;
			else
				part2++;
		}
		printf("%i  %i \n", part1, part2);


	}

	/* Store the number of nodes. */

	/*Free memory used by METIS*/
	free(xadj);
	free(adjncy);
	free(vwgt);
	free(adjwgt);
	free(nvtxs);
	free(ncon);
	free(nparts);
	free(objval);
	free(particle);
	free(options);
	printf("Successfully split the space\n");

	/* Call it a day. */
	return engine_err_ok;

	/* Bisect recursively */
	/* Interior, recursive function that actually does the split. */
	int engine_split_bisect_rec( int N_min , int N_max , int x_min , int x_max , int y_min , int y_max , int z_min , int z_max , int flags) {

		int i, j, k, m, Nm;
		int hx, hy, hz;
		unsigned int flag = 0;
		struct space_cell *c;

		/* Check inputs. */
		if ( x_max < x_min || y_max < y_min || z_max < z_min )
			return error(engine_err_domain);

		/* Is there nothing left to split? */
		if ( N_min == N_max ) {

			/* Flag as ghost or not? */
			if( flags == engine_split_MPI )
			{
				if ( N_min != e->nodeID )
					flag = cell_flag_ghost;

				/* printf("engine_split_bisect: marking range [ %i..%i , %i..%i , %i..%i ] with flag %i.\n",
                x_min, x_max, y_min, y_max, z_min, z_max, flag ); */

				/* Run through the cells. */
				for ( i = x_min ; i < x_max ; i++ )
					for ( j = y_min ; j < y_max ; j++ )
						for ( k = z_min ; k < z_max ; k++ ) {
							c = &( e->s.cells[ space_cellid(&(e->s),i,j,k) ] );
							c->flags |= flag;
							c->nodeID = N_min;
						}
			}else{

				for ( i = x_min ; i < x_max ; i++ )
					for ( j = y_min ; j < y_max ; j++ )
						for ( k = z_min ; k < z_max ; k++ ) {
							c = &( e->s.cells[ space_cellid(&(e->s),i,j,k) ] );
							c->GPUID = N_min;
						}
			}
		}

		/* Otherwise, bisect. */
		else {

			hx = x_max - x_min;
			hy = y_max - y_min;
			hz = z_max - z_min;
			Nm = (N_min + N_max) / 2;

			/* Is the x-axis the largest? */
					if ( hx > hy && hx > hz ) {
						m = (x_min + x_max) / 2;
						if ( engine_split_bisect_rec( N_min , Nm , x_min , m , y_min , y_max , z_min , z_max , flags) < 0 ||
								engine_split_bisect_rec( Nm+1 , N_max , m , x_max , y_min , y_max , z_min , z_max , flags) < 0 )
							return error(engine_err);
					}

					/* Nope, maybe the y-axis? */
					else if ( hy > hz ) {
						m = (y_min + y_max) / 2;
						if ( engine_split_bisect_rec( N_min , Nm , x_min , x_max , y_min , m , z_min , z_max , flags ) < 0 ||
								engine_split_bisect_rec( Nm+1 , N_max , x_min , x_max , m , y_max , z_min , z_max , flags) < 0 )
							return error(engine_err);
					}

					/* Then it has to be the z-axis. */
					else {
						m = (z_min + z_max) / 2;
						if ( engine_split_bisect_rec( N_min , Nm , x_min , x_max , y_min , y_max , z_min , m , flags) < 0 ||
								engine_split_bisect_rec( Nm+1 , N_max , x_min , x_max , y_min , y_max , m , z_max , flags) < 0 )
							return error(engine_err);
					}

		}

		/* So far, so good! */
		return engine_err_ok;

	}

	/* Check inputs. */
	if ( e == NULL )
		return error(engine_err_null);

	/* Call the recursive bisection. */
	if ( engine_split_bisect_rec( 0 , N-1 , 0 , e->s.cdim[0] , 0 , e->s.cdim[1] , 0 , e->s.cdim[2] , flags) < 0 )
		return error(engine_err);

	/* Store the number of nodes. */
	e->nr_nodes = N;

	/* Call it a day. */
	return engine_err_ok;

}

#endif

/* Interior, recursive function that actually does the split. */
static int engine_split_bisect_rec(struct engine *e, int N_min , int N_max ,
		int x_min , int x_max , int y_min , int y_max , int z_min , int z_max ) {

	int i, j, k, m, Nm;
	int hx, hy, hz;
	unsigned int flag = 0;
	struct space_cell *c;

	/* Check inputs. */
	if ( x_max < x_min || y_max < y_min || z_max < z_min )
		return error(engine_err_domain);

	/* Is there nothing left to split? */
	if ( N_min == N_max ) {

		/* Flag as ghost or not? */
		if ( N_min != e->nodeID )
			flag = cell_flag_ghost;

		/* printf("engine_split_bisect: marking range [ %i..%i , %i..%i , %i..%i ] with flag %i.\n",
            x_min, x_max, y_min, y_max, z_min, z_max, flag ); */

		/* Run through the cells. */
		for ( i = x_min ; i < x_max ; i++ )
			for ( j = y_min ; j < y_max ; j++ )
				for ( k = z_min ; k < z_max ; k++ ) {
					c = &( e->s.cells[ space_cellid(&(e->s),i,j,k) ] );
					c->flags |= flag;
					c->nodeID = N_min;
				}
	}

	/* Otherwise, bisect. */
	else {

		hx = x_max - x_min;
		hy = y_max - y_min;
		hz = z_max - z_min;
		Nm = (N_min + N_max) / 2;

		/* Is the x-axis the largest? */
		if ( hx > hy && hx > hz ) {
			m = (x_min + x_max) / 2;
			if ( engine_split_bisect_rec(e, N_min , Nm , x_min , m , y_min , y_max , z_min , z_max ) < 0 ||
					engine_split_bisect_rec(e, Nm+1 , N_max , m , x_max , y_min , y_max , z_min , z_max ) < 0 )
				return error(engine_err);
		}

		/* Nope, maybe the y-axis? */
		else if ( hy > hz ) {
			m = (y_min + y_max) / 2;
			if ( engine_split_bisect_rec(e, N_min , Nm , x_min , x_max , y_min , m , z_min , z_max ) < 0 ||
					engine_split_bisect_rec(e, Nm+1 , N_max , x_min , x_max , m , y_max , z_min , z_max ) < 0 )
				return error(engine_err);
		}

		/* Then it has to be the z-axis. */
		else {
			m = (z_min + z_max) / 2;
			if ( engine_split_bisect_rec(e, N_min , Nm , x_min , x_max , y_min , y_max , z_min , m ) < 0 ||
					engine_split_bisect_rec(e, Nm+1 , N_max , x_min , x_max , y_min , y_max , m , z_max ) < 0 )
				return error(engine_err);
		}

	}

	/* So far, so good! */
	return engine_err_ok;

}




/**
 * @brief Split the computational domain over a number of nodes using
 *      bisection.
 *
 * @param e The #engine to split up.
 * @param N The number of computational nodes.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */

int engine_split_bisect ( struct engine *e , int N ) {

	/* Check inputs. */
	if ( e == NULL )
		return error(engine_err_null);

	/* Call the recursive bisection. */
	if ( engine_split_bisect_rec(e, 0 , N-1 , 0 , e->s.cdim[0] , 0 , e->s.cdim[1] , 0 , e->s.cdim[2] ) < 0 )
		return error(engine_err);

	/* Store the number of nodes. */
	e->nr_nodes = N;

	/* Call it a day. */
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

int engine_setexplepot ( struct engine *e , struct potential *ep ) {

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

	struct particle *p;
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
				type[count] = p->type;
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

	struct particle *p;
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
				type[count] = p->type;
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

	struct particle *p;
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
		for ( k = c->count-1 ; k >= 0 && !(c->parts[k].flags & PARTICLE_FLAG_GHOST) ; k-- ) {

			/* Get a hold of the particle. */
			p = &( c->parts[k] );
			if ( p->flags & PARTICLE_FLAG_GHOST )
				continue;

			/* get this particle's data, where requested. */
			if ( x != NULL )
				for ( j = 0 ; j < 3 ; j++ )
					x[count*3+j] = c->origin[j] + p->x[j];
			if ( v != NULL)
				for ( j = 0 ; j < 3 ; j++ )
					v[count*3+j] = p->v[j];
			if ( type != NULL )
				type[count] = p->type;
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

	struct particle p;
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
	p.flags = PARTICLE_FLAG_NONE;

	/* loop over the entries. */
	for ( j = 0 ; j < N ; j++ ) {

		/* set the particle data. */
		p.type = type[j];
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
		if ( space_addpart( s , &p , &x[3*j] ) < 0 )
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

	struct particle p;
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
	p.flags = PARTICLE_FLAG_GHOST;

	/* loop over the entries. */
	for ( j = 0 ; j < N ; j++ ) {

		/* set the particle data. */
		p.type = type[j];
		if ( pid != NULL )
			p.id = pid[j];
		else
			p.id = j;
		if ( vid != NULL )
			p.vid = vid[j];
		if ( flags != NULL )
			p.flags = flags[j] | PARTICLE_FLAG_GHOST;
		if ( v != NULL )
			for ( k = 0 ; k < 3 ; k++ )
				p.v[k] = v[j*3+k];
		if ( q != 0 )
			p.q = q[j];

		/* add the part to the space. */
		if ( space_addpart( s , &p , &x[3*j] ) < 0 )
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

int engine_addtype ( struct engine *e , double mass , double charge , char *name , char *name2 ) {

	/* check for nonsense. */
	if ( e == NULL )
		return error(engine_err_null);
	if ( e->nr_types >= e->max_type )
		return error(engine_err_range);

	/* set the type. */
	e->types[e->nr_types].mass = mass;
	e->types[e->nr_types].imass = 1.0 / mass;
	e->types[e->nr_types].charge = charge;
	if ( name != NULL )
		strcpy( e->types[e->nr_types].name , name );
	else
		strcpy( e->types[e->nr_types].name , "X" );
	if ( name2 != NULL )
		strcpy( e->types[e->nr_types].name2 , name2 );
	else
		strcpy( e->types[e->nr_types].name2 , "X" );

	/* bring good tidings. */
	return e->nr_types++;

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

int engine_addpot ( struct engine *e , struct potential *p , int i , int j ) {

	/* check for nonsense. */
	if ( e == NULL )
		return error(engine_err_null);
	if ( i < 0 || i >= e->max_type || j < 0 || j >= e->max_type )
		return error(engine_err_range);

	/* store the potential. */
	e->p[ i * e->max_type + j ] = p;
	if ( i != j )
		e->p[ j * e->max_type + i ] = p;

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
	struct particle *p;
	struct space *s = &e->s;

	/* Is MPI really needed? */
	if ( e->flags & engine_flag_mpi && e->nr_nodes == 1 )
		e->flags &= ~( engine_flag_mpi | engine_flag_async );

#ifdef WITH_MPI
	/* Set up async communication? */
	if ( e->flags & engine_flag_async ) {

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
 * @brief Update the particle velocities and positions, re-shuffle if
 *      appropriate.
 * @param e The #engine on which to run.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */

int engine_advance ( struct engine *e ) {

	int cid, pid, k, delta[3], step;
	struct space_cell *c, *c_dest;
	struct particle *p;
	struct space *s;
	FPTYPE dt, w, h[3];
	double epot = 0.0, epot_local;

	/* Get a grip on the space. */
	s = &(e->s);
	dt = e->dt;
	for ( k = 0 ; k < 3 ; k++ )
		h[k] = s->h[k];

	/* update the particle velocities and positions */
	if ( e->flags & engine_flag_verlet || e->flags & engine_flag_mpi ) {

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
					w = dt * e->types[p->type].imass;
					for ( k = 0 ; k < 3 ; k++ ) {
						p->v[k] += p->f[k] * w;
						p->x[k] += dt * p->v[k];
					}
				}
			}
#pragma omp atomic
epot += epot_local;
		}

	}
	else {

		/* Collect potential energy from ghosts. */
		for ( cid = 0 ; cid < s->nr_ghost ; cid++ )
			epot += s->cells[ s->cid_ghost[cid] ].epot;

#pragma omp parallel private(cid,c,pid,p,w,k,delta,c_dest,epot_local)
		{
			step = omp_get_num_threads(); epot_local = 0.0;
			for ( cid = omp_get_thread_num() ; cid < s->nr_real ; cid += step ) {
				c = &(s->cells[ s->cid_real[cid] ]);
				epot_local += c->epot;
				pid = 0;
				while ( pid < c->count ) {

					p = &( c->parts[pid] );
					w = dt * e->types[p->type].imass;
					for ( k = 0 ; k < 3 ; k++ ) {
						p->v[k] += p->f[k] * w;
						p->x[k] += dt * p->v[k];
						delta[k] = __builtin_isgreaterequal( p->x[k] , h[k] ) - __builtin_isless( p->x[k] , 0.0 );
					}

					/* do we have to move this particle? */
							if ( ( delta[0] != 0 ) || ( delta[1] != 0 ) || ( delta[2] != 0 ) ) {
								for ( k = 0 ; k < 3 ; k++ )
									p->x[k] -= delta[k] * h[k];
								c_dest = &( s->cells[ space_cellid( s ,
										(c->loc[0] + delta[0] + s->cdim[0]) % s->cdim[0] ,
										(c->loc[1] + delta[1] + s->cdim[1]) % s->cdim[1] ,
										(c->loc[2] + delta[2] + s->cdim[2]) % s->cdim[2] ) ] );

								pthread_mutex_lock(&c_dest->cell_mutex);
								space_cell_add_incomming( c_dest , p );
								pthread_mutex_unlock(&c_dest->cell_mutex);

								s->celllist[ p->id ] = c_dest;
								c->count -= 1;
								if ( pid < c->count ) {
									c->parts[pid] = c->parts[c->count];
									s->partlist[ c->parts[pid].id ] = &( c->parts[pid] );
								}
							}
							else
								pid += 1;
				}
			}
#pragma omp atomic
			epot += epot_local;
		}

		/* Welcome the new particles in each cell. */
#pragma omp parallel for schedule(static)
		for ( cid = 0 ; cid < s->nr_marked ; cid++ )
			space_cell_welcome( &(s->cells[ s->cid_marked[cid] ]) , s->partlist );

	}

	/* Store the accumulated potential energy. */
	s->epot_nonbond += epot;
	s->epot += epot;

	/* return quietly */
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

	ticks tic, tic_step = getticks();

	/* increase the time stepper */
	e->time += 1;

	/* prepare the space */
	tic = getticks();
	if ( space_prepare( &e->s ) != space_err_ok )
		return error(engine_err_space);
	e->timers[engine_timer_prepare] += getticks() - tic;

	/* Make sure the verlet lists are up to date. */
	if ( e->flags & engine_flag_verlet ) {

		/* Start the clock. */
		tic = getticks();

		/* Check particle movement and update cells if necessary. */
		if ( engine_verlet_update( e ) < 0 )
			return error(engine_err);

		/* Store the timing. */
		e->timers[engine_timer_verlet] += getticks() - tic;

	}

	/* Otherwise, if async MPI, move the particles accross the
       node boundaries. */
	else { // if ( e->flags & engine_flag_async ) {
		tic = getticks();
		if ( engine_shuffle( e ) < 0 )
			return error(engine_err_space);
		e->timers[engine_timer_advance] += getticks() - tic;
	}

#ifdef WITH_MPI
	/* Re-distribute the particles to the processors. */
	if ( e->flags & engine_flag_mpi ) {

		/* Start the clock. */
		tic = getticks();

		if ( e->flags & engine_flag_async ) {
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
#if defined(HAVE_CUDA) && defined(WITH_CUDA)
	if ( e->flags & engine_flag_cuda ) {
		if ( engine_nonbond_cuda( e ) < 0 )
			return error(engine_err);
	}
	else
#endif
if ( engine_nonbond_eval( e ) < 0 )
	return error(engine_err);
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

	/* update the particle velocities and positions. */
	tic = getticks();
	if ( engine_advance( e ) < 0 )
		return error(engine_err);
	e->timers[engine_timer_advance] += getticks() - tic;

	/* Shake the particle positions? */
	if ( e->nr_rigids > 0 ) {

#ifdef WITH_MPI
		/* If we have to do some communication first... */
		if ( e->flags & engine_flag_mpi ) {

			/* Sort the constraints. */
			tic = getticks();
			if ( engine_rigid_sort( e ) != 0 )
				return error(engine_err);
			e->timers[engine_timer_rigid] += getticks() - tic;

			/* Start the clock. */
			tic = getticks();

			if ( e->flags & engine_flag_async ) {
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

	/* return quietly */
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
int engine_init_mpi ( struct engine *e , const double *origin , const double *dim , double *L , double cutoff , unsigned int period , int max_type , unsigned int flags , MPI_Comm comm , int rank ) {

	/* Init the engine. */
	if ( engine_init( e , origin , dim , L , cutoff , period , max_type , flags | engine_flag_mpi ) < 0 )
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
		for ( j = 0 ; j < e->nr_types ; j++ )
			for ( k = j ; k < e->nr_types ; k++ ) {
				if ( e->p[ j*e->max_type + k ] != NULL )
					potential_clear( e->p[ j*e->max_type + k ] );
				if ( e->p_bond[ j*e->max_type + k ] != NULL )
					potential_clear( e->p_bond[ j*e->max_type + k ] );
			}
		for ( k = 0 ; k < e->nr_anglepots ; k++ )
			potential_clear( e->p_angle[k] );
		for ( k = 0 ; k < e->nr_dihedralpots ; k++ )
			potential_clear( e->p_dihedral[k] );
		free( e->p );
	}
	if ( e->p_bond != NULL )
		free( e->p_bond );
	if ( e->p_angle != NULL )
		free( e->p_angle );
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


/**
 * @brief Initialize an #engine with the given data.
 *
 * @param e The #engine to initialize.
 * @param origin An array of three doubles containing the cartesian origin
 *      of the space.
 * @param dim An array of three doubles containing the size of the space.
 * @param L The minimum cell edge length in each dimension.
 * @param cutoff The maximum interaction cutoff to use.
 * @param period A bitmask describing the periodicity of the domain
 *      (see #space_periodic_full).
 * @param max_type The maximum number of particle types that will be used
 *      by this engine.
 * @param flags Bit-mask containing the flags for this engine.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */

int engine_init ( struct engine *e , const double *origin , const double *dim , double *L ,
		double cutoff , unsigned int period , int max_type , unsigned int flags ) {

	int cid;

	/* make sure the inputs are ok */
	if ( e == NULL || origin == NULL || dim == NULL || L == NULL )
		return error(engine_err_null);

	/* Check for bad flags. */
#ifdef FPTYPE_DOUBLE
	if ( e->flags & engine_flag_cuda )
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
	if ( ( e->bonds = (struct bond *)malloc( sizeof( struct bond ) * e->bonds_size ) ) == NULL )
		return error(engine_err_malloc);
	e->nr_bonds = 0;

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
	if ( ( e->angles = (struct angle *)malloc( sizeof( struct angle ) * e->angles_size ) ) == NULL )
		return error(engine_err_malloc);
	e->nr_angles = 0;

	/* Init the dihedrals array. */
	e->dihedrals_size = 100;
	if ( ( e->dihedrals = (struct dihedral *)malloc( sizeof( struct dihedral ) * e->dihedrals_size ) ) == NULL )
		return error(engine_err_malloc);
	e->nr_dihedrals = 0;

	/* set the maximum nr of types */
	if ( flags & engine_flag_nullpart )
		max_type += 1;
	e->max_type = max_type;
	e->nr_types = 0;
	if ( ( e->types = (struct particle_type *)malloc( sizeof(struct particle_type) * max_type ) ) == NULL )
		return error(engine_err_malloc);
	if ( flags & engine_flag_nullpart ) {
		e->types[0].id = 0;
		e->types[0].mass = 0.0;
		e->types[0].imass = 0.0;
		e->types[0].charge = 0.0;
		e->types[0].eps = 0.0;
		e->types[0].rmin = 0.0;
		strcpy( e->types[0].name , "NULL" );
		strcpy( e->types[0].name2 , "NULL" );
		e->nr_types = 1;
	}

	/* Init the sets. */
	e->sets = NULL;
	e->nr_sets = 0;

	/* allocate the interaction matrices */
	if ( ( e->p = (struct potential **)malloc( sizeof(struct potential *) * max_type * max_type ) ) == NULL )
		return error(engine_err_malloc);
	bzero( e->p , sizeof(struct potential *) * max_type * max_type );
	if ( (e->p_bond = (struct potential **)malloc( sizeof(struct potential *) * max_type * max_type )) == NULL)
		return error(engine_err_malloc);
	bzero( e->p_bond , sizeof(struct potential *) * max_type * max_type );
	e->anglepots_size = 100;
	if ( (e->p_angle = (struct potential **)malloc( sizeof(struct potential *) * e->anglepots_size )) == NULL)
		return error(engine_err_malloc);
	bzero( e->p_angle , sizeof(struct potential *) * e->anglepots_size );
	e->nr_anglepots = 0;
	e->dihedralpots_size = 100;
	if ( (e->p_dihedral = (struct potential **)malloc( sizeof(struct potential *) * e->dihedralpots_size )) == NULL)
		return error(engine_err_malloc);
	bzero( e->p_dihedral , sizeof(struct potential *) * e->dihedralpots_size );
	e->nr_dihedralpots = 0;

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

			/* all is well... */
			return engine_err_ok;

}
