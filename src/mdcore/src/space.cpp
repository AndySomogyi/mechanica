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
#include <string.h>
#include <pthread.h>
#include <math.h>

/* Include conditional headers. */
#include "mdcore_config.h"
#ifdef HAVE_OPENMP
#include <omp.h>
#endif
#ifdef WITH_MPI
#include <mpi.h>
#endif

/* include local headers */
#include "errs.h"
#include "fptype.h"
#include "lock.h"
#include <MxParticle.h>
#include <space_cell.h>
#include "task.h"
#include "space.h"
#include "engine.h"
#include "../../rendering/NOMStyle.hpp"
#include <iostream>

#pragma clang diagnostic ignored "-Wwritable-strings"


/* the last error */
int space_err = space_err_ok;


/* the error macro. */
#define error(id) ( space_err = errs_register( id , space_err_msg[-(id)] , __LINE__ , __FUNCTION__ , __FILE__ ) )

/* list of error messages. */
const char *space_err_msg[10] = {
        "Nothing bad happened.",
        "An unexpected NULL pointer was encountered.",
        "A call to malloc failed, probably due to insufficient memory.",
        "An error occured when calling a cell function.",
        "A call to a pthread routine failed.",
        "One or more values were outside of the allowed range.",
        "Too many pairs associated with a single particle in Verlet list.",
        "Task list too short.",
        "An error occured when calling a task function.",
        "Invalid particle id"
};


/** 
 * @brief Get the sort-ID and flip the cells if necessary.
 *
 * @param s The #space.
 * @param ci Double pointer to the first #cell.
 * @param cj Double pointer to the second #cell.
 * 
 * @return The sort ID of both cells, which may be swapped.
 */

int space_getsid ( struct space *s , struct space_cell **ci ,
        struct space_cell **cj , FPTYPE *shift ) {

    int k, sid;
    struct space_cell *temp;
    FPTYPE lshift[3];

    /* Shift vector provided? */
    if ( shift == NULL )
        shift = lshift;

    /* Compute the shift. */
    for ( k = 0 ; k < 3 ; k++ ) {
        shift[k] = (*cj)->origin[k] - (*ci)->origin[k];
        if ( shift[k] * 2 > s->dim[k] )
            shift[k] -= s->dim[k];
        else if ( shift[k] * 2 < -s->dim[k] )
            shift[k] += s->dim[k];
    }

    /* Get the ID of the sortlist for this shift. */
    for ( sid = 0 , k = 0 ; k < 3 ; k++ )
        sid = 3*sid + ( (shift[k] < 0) ? 0 : ( (shift[k] > 0) ? 2 : 1 ) );

    /* Flip the cells around? */
    if ( cell_flip[sid] ) {
        temp = *ci; *ci = *cj; *cj = temp;
        shift[0] = -shift[0];
        shift[1] = -shift[1];
        shift[2] = -shift[2];
    }

    /* Return the flipped sort ID. */
    return cell_sortlistID[sid];

}


/**
 * @brief Clear all particles from the ghost cells in this #space.
 *
 * @param s The #space to flush.
 *
 * @return #space_err_ok or < 0 on error (see #space_err).
 */

int space_flush_ghosts ( struct space *s ) {

    int cid;

    /* check input. */
    if ( s == NULL )
        return error(space_err_null);

    /* loop through the cells. */
    for ( cid = 0 ; cid < s->nr_cells ; cid++ )
        if ( s->cells[cid].flags & cell_flag_ghost ) {
            s->nr_parts -= s->cells[cid].count;
            s->cells[cid].count = 0;
        }

    /* done for now. */
    return space_err_ok;

}


/**
 * @brief Clear all particles from this #space.
 *
 * @param s The #space to flush.
 *
 * @return #space_err_ok or < 0 on error (see #space_err).
 */

int space_flush ( struct space *s ) {

    int cid;

    /* check input. */
    if ( s == NULL )
        return error(space_err_null);

    /* loop through the cells. */
    for ( cid = 0 ; cid < s->nr_cells ; cid++ )
        s->cells[cid].count = 0;

    /* Set the nr of parts to zero. */
    s->nr_parts = 0;

    /* done for now. */
    return space_err_ok;

}


/**
 * @brief Prepare the space before a time step.
 *
 * @param s A pointer to the #space to prepare.
 *
 * @return #space_err_ok or < 0 on error (see #space_err)
 *
 * Initializes a #space for a single time step. This routine runs
 * through the particles and sets their forces to zero.
 */

int space_prepare ( struct space *s ) {

    int pid, cid, j, k;

    /* re-set some counters. */
    s->nr_swaps = 0;
    s->nr_stalls = 0;
    s->epot = 0.0;
    s->epot_nonbond = 0.0;
    s->epot_bond = 0.0;
    s->epot_angle = 0.0;
    s->epot_dihedral = 0.0;
    s->epot_exclusion = 0.0;

    /* Run through the tasks and set the waits. */
    for ( k = 0 ; k < s->nr_tasks ; k++ )
        for ( j = 0 ; j < s->tasks[k].nr_unlock ; j++ )
            s->tasks[k].unlock[j]->wait += 1;

    /* run through the cells and re-set the potential energy and forces */
    for ( j = 0 ; j < s->nr_marked ; j++ ) {
        cid = s->cid_marked[j];
        s->cells[cid].epot = 0.0;
        if ( s->cells[cid].flags & cell_flag_ghost )
            continue;
        for ( pid = 0 ; pid < s->cells[cid].count ; pid++ )
            for ( k = 0 ; k < 3 ; k++ )
                s->cells[cid].parts[pid].f[k] = 0.0;
    }

    /* what else could happen? */
    return space_err_ok;

}


/**
 * @brief Run through the cells of a #space and make sure every particle is in
 * its place.
 *
 * @param s The #space on which to operate.
 *
 * @returns #space_err_ok or < 0 on error.
 *
 * Runs through the cells of @c s and if a particle has stepped outside the
 * cell bounds, moves it to the correct cell.
 */
/* TODO: Check non-periodicity and ghost cells. */

int space_shuffle ( struct space *s ) {

    int k, cid, pid, delta[3];
    FPTYPE h[3];
    struct space_cell *c, *c_dest;
    struct MxParticle *p;

    /* Get a local copy of h. */
    for ( k = 0 ; k < 3 ; k++ )
        h[k] = s->h[k];

#pragma omp parallel for schedule(static), private(cid,c,pid,p,k,delta,c_dest)
    for ( cid = 0 ; cid < s->nr_marked ; cid++ ) {
        c = &(s->cells[ s->cid_marked[cid] ]);
        pid = 0;
        while ( pid < c->count ) {

            p = &( c->parts[pid] );
            for ( k = 0 ; k < 3 ; k++ )
                delta[k] = __builtin_isgreaterequal( p->x[k] , h[k] ) - __builtin_isless( p->x[k] , 0.0 );

            /* do we have to move this particle? */
            if ( ( delta[0] != 0 ) || ( delta[1] != 0 ) || ( delta[2] != 0 ) ) {
                for ( k = 0 ; k < 3 ; k++ ) {
                    p->x[k] -= delta[k] * h[k];
                    p->p0[k] -= delta[k] *h[k];
                }

                c_dest = &( s->cells[ space_cellid( s ,
                        (c->loc[0] + delta[0] + s->cdim[0]) % s->cdim[0] ,
                        (c->loc[1] + delta[1] + s->cdim[1]) % s->cdim[1] ,
                        (c->loc[2] + delta[2] + s->cdim[2]) % s->cdim[2] ) ] );

                if ( c_dest->flags & cell_flag_marked ) {
                    pthread_mutex_lock(&c_dest->cell_mutex);
                    space_cell_add_incomming( c_dest , p );
                    pthread_mutex_unlock(&c_dest->cell_mutex);
                    s->celllist[ p->id ] = c_dest;
                }
                else {
                    s->partlist[ p->id ] = NULL;
                    s->celllist[ p->id ] = NULL;
                }

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

    /* all is well... */
    return space_err_ok;

}


/**
 * @brief Run through the non-ghost cells of a #space and make sure every
 * particle is in its place.
 *
 * @param s The #space on which to operate.
 *
 * @returns #space_err_ok or < 0 on error.
 *
 * Runs through the cells of @c s and if a particle has stepped outside the
 * cell bounds, moves it to the correct cell.
 */
/* TODO: Check non-periodicity and ghost cells. */

int space_shuffle_local ( struct space *s ) {

    int k, cid, pid, delta[3];
    FPTYPE h[3];
    struct space_cell *c, *c_dest;
    struct MxParticle *p;

    /* Get a local copy of h. */
    for ( k = 0 ; k < 3 ; k++ )
        h[k] = s->h[k];

#pragma omp parallel for schedule(static), private(cid,c,pid,p,k,delta,c_dest)
    for ( cid = 0 ; cid < s->nr_real ; cid++ ) {
        c = &(s->cells[ s->cid_real[cid] ]);
        pid = 0;
        while ( pid < c->count ) {

            p = &( c->parts[pid] );
            for ( k = 0 ; k < 3 ; k++ )
                delta[k] = __builtin_isgreaterequal( p->x[k] , h[k] ) - __builtin_isless( p->x[k] , 0.0 );

            /* do we have to move this particle? */
            if ( ( delta[0] != 0 ) || ( delta[1] != 0 ) || ( delta[2] != 0 ) ) {

                for ( k = 0 ; k < 3 ; k++ ) {
                    p->x[k] -= delta[k] * h[k];
                    p->p0[k] -= delta[k] * h[k];
                }

                c_dest = &( s->cells[ space_cellid( s ,
                        (c->loc[0] + delta[0] + s->cdim[0]) % s->cdim[0] ,
                        (c->loc[1] + delta[1] + s->cdim[1]) % s->cdim[1] ,
                        (c->loc[2] + delta[2] + s->cdim[2]) % s->cdim[2] ) ] );

                if ( c_dest->flags & cell_flag_marked ) {
                    pthread_mutex_lock(&c_dest->cell_mutex);
                    space_cell_add_incomming( c_dest , p );
                    pthread_mutex_unlock(&c_dest->cell_mutex);
                    s->celllist[ p->id ] = c_dest;
                }
                else {
                    s->partlist[ p->id ] = NULL;
                    s->celllist[ p->id ] = NULL;
                }
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

    /* all is well... */
    return space_err_ok;

}



int space_addpart ( struct space *s , struct MxParticle *p , double *x, struct MxParticle **result ) {

    int k, ind[3];
    struct MxParticle **temp;
    struct space_cell **tempc, *c;


    /* check input */
    if ( s == NULL || p == NULL || x == NULL )
        return error(space_err_null);

    /* do we need to extend the partlist? */
    if ( s->nr_parts == s->size_parts ) {
        s->size_parts += space_partlist_incr;
        if ( ( temp = (struct MxParticle **)malloc( sizeof(struct MxParticle *) * s->size_parts ) ) == NULL )
            return error(space_err_malloc);
        if ( ( tempc = (struct space_cell **)malloc( sizeof(struct space_cell *) * s->size_parts ) ) == NULL )
            return error(space_err_malloc);
        memcpy( temp , s->partlist , sizeof(struct MxParticle *) * s->nr_parts );
        memcpy( tempc , s->celllist , sizeof(struct space_cell *) * s->nr_parts );
        free( s->partlist );
        free( s->celllist );
        s->partlist = temp;
        s->celllist = tempc;
    }

    /* Increase the number of parts. */
    s->nr_parts++;
    
    if(p->id < 0 || p->id >= s->nr_parts) {
        return error(space_err_invalid_partid);
    }
    
    /* get the hypothetical cell coordinate */
    for ( k = 0 ; k < 3 ; k++ )
        ind[k] = (x[k] - s->origin[k]) * s->ih[k];
    
    /* is this particle within the space? */
    for ( k = 0 ; k < 3 ; k++ )
        if ( ind[k] < 0 || ind[k] >= s->cdim[k] )
            return error(space_err_range);
    
    // treat large particles in the large parts cell
    if(p->flags & PARTICLE_LARGE) {
        std::cout << "adding large particle: " << p->id << std::endl;
        c = &s->largeparts;
    }
    else {
        /* get the appropriate cell */
        c = &( s->cells[ space_cellid(s,ind[0],ind[1],ind[2]) ] );
    }

    /* make the particle position local */
    for ( k = 0 ; k < 3 ; k++ )
        p->x[k] = x[k] - c->origin[k];

    /* delegate the particle to the cell */
    if ( ( s->partlist[p->id] = space_cell_add( c , p , s->partlist ) ) == NULL )
        return error(space_err_cell);
    
    s->celllist[p->id] = c;
    
    if(result) {
        *result = s->partlist[p->id];
    }
    
    MxParticleType *type = &_Engine.types[p->typeId];
    NOMStyle *style = p->style ? p->style : type->style;
    
    if(style->flags & STYLE_VISIBLE) {
        if(p->flags & PARTICLE_LARGE) {
            s->nr_visable_large_parts++;
        }
        else {
            s->nr_visable_parts++;
        }
    }

    /* end well */
    return space_err_ok;
}


/**
 * @brief Get the absolute position of a particle
 *
 * @param s The #space in which the particle resides.
 * @param id The local id of the #part.
 * @param x A pointer to a vector of at least three @c doubles in
 *      which to store the particle position.
 *
 */

int space_getpos ( struct space *s , int id , FPTYPE *x ) {

    int k;

    /* Sanity check. */
    if ( s == NULL || x == NULL )
        return error(space_err_null);
    if ( id >= s->nr_parts )
        return error(space_err_range);

    /* Copy the position to x. */
    for ( k = 0 ; k < 3 ; k++ )
        x[k] = s->partlist[id]->x[k] + s->celllist[id]->origin[k];

    /* All is well... */
    return space_err_ok;

}

int space_setpos ( struct space *s , int id , FPTYPE *x ) {

    int k;

    /* Sanity check. */
    if ( s == NULL || x == NULL )
        return error(space_err_null);
    if ( id >= s->nr_parts )
        return error(space_err_range);

    /* Copy the position to x. */
    for ( k = 0 ; k < 3 ; k++ )
        s->partlist[id]->x[k] = x[k] - s->celllist[id]->origin[k];

    /* All is well... */
    return space_err_ok;

}


/**
 * @brief Add a task to the given space.
 *
 * @param s The #space.
 * @param type The task type.
 * @param subtype The task subtype.
 * @param flags The task flags.
 * @param i Index of the first cell/domain.
 * @param j Index of the second cell/domain.
 *
 * @return A pointer to the newly added #task or @c NULL if anything went wrong.
 */

struct task *space_addtask ( struct space *s , int type , int subtype , int flags , int i , int j ) {

    struct task *t = &s->tasks[ s->nr_tasks ];

    /* Is there enough space? */
    if ( s->nr_tasks >= s->tasks_size ) {
        error( space_err_nrtasks );
        return NULL;
    }

    /* Fill in the task data. */
    t->type = type;
    t->subtype = subtype;
    t->flags = flags;
    t->i = i;
    t->j = j;

    /* Init some other values. */
    t->wait = 0;
    t->nr_unlock = 0;

    /* Increase the task counter. */
    s->nr_tasks += 1;

    /* Sayonara, suckers! */
    return t;

}


/**
 * @brief Initialize the space with the given dimensions.
 *
 * @param s The #space to initialize.
 * @param origin Pointer to an array of three doubles specifying the origin
 *      of the rectangular domain.
 * @param dim Pointer to an array of three doubles specifying the length
 *      of the rectangular domain along each dimension.
 * @param L The minimum cell edge length, in each dimension.
 * @param cutoff A double-precision value containing the maximum cutoff lenght
 *      that will be used in the potentials.
 * @param period Unsigned integer containing the flags #space_periodic_x,
 *      #space_periodic_y and/or #space_periodic_z or #space_periodic_full.
 *
 * @return #space_err_ok or <0 on error (see #space_err).
 * 
 * This routine initializes the fields of the #space @c s, creates the cells and
 * generates the cell-pair list.
 */

int space_init ( struct space *s , const double *origin , const double *dim ,
        double *L , double cutoff , unsigned int period ) {

    int i, j, k, l[3], ii, jj, kk;
    int id1, id2, sid;
    double o[3], lh[3];
    struct space_cell *ci, *cj;

    /* check inputs */
    if ( s == NULL || origin == NULL || dim == NULL || L == NULL )
        return error(space_err_null);

    /* Clear the space. */
    bzero( s , sizeof(struct space) );

    /* set origin and compute the dimensions */
    for ( i = 0 ; i < 3 ; i++ ) {
        s->origin[i] = origin[i];
        s->dim[i] = dim[i];
        s->cdim[i] = floor( dim[i] / L[i] );
    }

    /* remember the cutoff */
    s->cutoff = cutoff;
    s->cutoff2 = cutoff*cutoff;

    /* set the periodicity */
    s->period = period;

    /* allocate the cells */
    s->nr_cells = s->cdim[0] * s->cdim[1] * s->cdim[2];
    s->cells = (struct space_cell *)malloc( sizeof(struct space_cell) * s->nr_cells );
    if ( s->cells == NULL )
        return error(space_err_malloc);

    /* get the dimensions of each cell */
    for ( i = 0 ; i < 3 ; i++ ) {
        s->h[i] = s->dim[i] / s->cdim[i];
        s->ih[i] = 1.0 / s->h[i];
    }
    /* initialize the cells  */
    for ( l[0] = 0 ; l[0] < s->cdim[0] ; l[0]++ ) {
        o[0] = origin[0] + l[0] * s->h[0];
        for ( l[1] = 0 ; l[1] < s->cdim[1] ; l[1]++ ) {
            o[1] = origin[1] + l[1] * s->h[1];
            for ( l[2] = 0 ; l[2] < s->cdim[2] ; l[2]++ ) {
                o[2] = origin[2] + l[2] * s->h[2];
                if ( space_cell_init( &(s->cells[space_cellid(s,l[0],l[1],l[2])]) , l , o , s->h ) < 0 )
                    return error(space_err_cell);
            }
        }
    }

    /* Make ghost layers if needed. */
    if ( s->period & space_periodic_ghost_x )
        for ( i = 0 ; i < s->cdim[0] ; i++ )
            for ( j = 0 ; j < s->cdim[1] ; j++ ) {
                s->cells[ space_cellid(s,i,j,0) ].flags |= cell_flag_ghost;
                s->cells[ space_cellid(s,i,j,s->cdim[2]-1) ].flags |= cell_flag_ghost;
            }
    if ( s->period & space_periodic_ghost_y )
        for ( i = 0 ; i < s->cdim[0] ; i++ )
            for ( j = 0 ; j < s->cdim[2] ; j++ ) {
                s->cells[ space_cellid(s,i,0,j) ].flags |= cell_flag_ghost;
                s->cells[ space_cellid(s,i,s->cdim[1]-1,j) ].flags |= cell_flag_ghost;
            }
    if ( s->period & space_periodic_ghost_z )
        for ( i = 0 ; i < s->cdim[1] ; i++ )
            for ( j = 0 ; j < s->cdim[2] ; j++ ) {
                s->cells[ space_cellid(s,0,i,j) ].flags |= cell_flag_ghost;
                s->cells[ space_cellid(s,s->cdim[0]-1,i,j) ].flags |= cell_flag_ghost;
            }

    /* Allocate buffers for the cid lists. */
    if ( ( s->cid_real = (int *)malloc( sizeof(int) * s->nr_cells ) ) == NULL ||
            ( s->cid_ghost = (int *)malloc( sizeof(int) * s->nr_cells ) ) == NULL ||
            ( s->cid_marked = (int *)malloc( sizeof(int) * s->nr_cells ) ) == NULL )
        return error(space_err_malloc);

    /* Fill the cid lists with marked, local and ghost cells. */
    s->nr_real = 0; s->nr_ghost = 0; s->nr_marked = 0;
    for ( k = 0 ; k < s->nr_cells ; k++ ) {
        s->cells[k].flags |= cell_flag_marked;
        s->cid_marked[ s->nr_marked++ ] = k;
        if ( s->cells[k].flags & cell_flag_ghost ) {
            s->cells[k].id = -s->nr_cells;
            s->cid_ghost[ s->nr_ghost++ ] = k;
        }
        else {
            s->cells[k].id = s->nr_real;
            s->cid_real[ s->nr_real++ ] = k;
        }
    }

    /* Get the span of the cells we will search for pairs. */
    for ( k = 0 ; k < 3 ; k++ )
        s->span[k] = ceil( cutoff * s->ih[k] );

    /* allocate the tasks array (pessimistic guess) */
    s->tasks_size = s->nr_cells * ( (2*s->span[0] + 1) * (2*s->span[1] + 1) * (2*s->span[2] + 1) + 1 );
    if ( ( s->tasks = (struct task *)malloc( sizeof(struct task) * s->tasks_size ) ) == NULL )
        return error(space_err_malloc);

    /* fill the cell pairs array */
    s->nr_tasks = 0;
    /* for every cell */
    for ( i = 0 ; i < s->cdim[0] ; i++ ) {
        for ( j = 0 ; j < s->cdim[1] ; j++ ) {
            for ( k = 0 ; k < s->cdim[2] ; k++ ) {

                /* get this cell's id */
                id1 = space_cellid(s,i,j,k);

                /* if this cell is a ghost cell, skip it. */
                if ( s->cells[id1].flags & cell_flag_ghost )
                    continue;

                /* for every neighbouring cell in the x-axis... */
                for ( l[0] = -s->span[0] ; l[0] <= s->span[0] ; l[0]++ ) {

                    /* get coords of neighbour */
                    ii = i + l[0];

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
                    for ( l[1] = -s->span[1] ; l[1] <= s->span[1] ; l[1]++ ) {

                        /* get coords of neighbour */
                        jj = j + l[1];

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
                        for ( l[2] = -s->span[2] ; l[2] <= s->span[2] ; l[2]++ ) {

                            /* Are these cells within the cutoff of each other? */
                            lh[0] = s->h[0]*fmax( abs(l[0])-1 , 0 );
                            lh[1] = s->h[1]*fmax( abs(l[1])-1 , 0 );
                            lh[2] = s->h[2]*fmax( abs(l[2])-1 , 0 );
                            if ( lh[0]*lh[0] + lh[1]*lh[1] + lh[2]*lh[2] > s->cutoff2 )
                                continue;

                            /* get coords of neighbour */
                            kk = k + l[2];

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

                            /* get the neighbour's id */
                            id2 = space_cellid(s,ii,jj,kk);

                            /* Get the pair sortID. */
                            ci = &s->cells[id1];
                            cj = &s->cells[id2];
                            sid = space_getsid( s , &ci , &cj , NULL );

                            /* store this pair? */
                            if ( id1 < id2 ||
                                    ( id1 == id2 && l[0] == 0 && l[1] == 0 && l[2] == 0 ) ||
                                    (s->cells[id2].flags & cell_flag_ghost ) ) {
                                if ( space_addtask(s, ( id1 == id2 ) ? task_type_self : task_type_pair,
                                                   task_subtype_none, sid, ci - s->cells,
                                                   cj - s->cells ) == NULL ) {
                                    return error(space_err);
                                }
                            }
                            
                        } /* for every neighbouring cell in the z-axis... */
                    } /* for every neighbouring cell in the y-axis... */
                } /* for every neighbouring cell in the x-axis... */

            }
        }
    }

    /* Run through the cells and add a sort task to each one. */
    for ( k = 0 ; k < s->nr_cells ; k++ ) {
        if ((s->cells[k].sort = space_addtask(s, task_type_sort, task_subtype_none, 0, k, -1)) == NULL) {
            return error(space_err);
        }
    }

    /* Run through the tasks and make each pair depend on the sorts. 
       Also set the flags for each sort. */
    for ( k = 0 ; k < s->nr_tasks ; k++ ) {
        if ( s->tasks[k].type == task_type_pair ) {
            if (task_addunlock( s->cells[ s->tasks[k].i ].sort , &s->tasks[k] ) != 0 ||
                task_addunlock( s->cells[ s->tasks[k].j ].sort , &s->tasks[k] ) != 0 ) {
                return error(space_err_task);
            }
            s->cells[ s->tasks[k].i ].sort->flags |= 1 << s->tasks[k].flags;
            s->cells[ s->tasks[k].j ].sort->flags |= 1 << s->tasks[k].flags;
        }
    }

    /* allocate and init the taboo-list */
    if ( (s->cells_taboo = (char *)malloc( sizeof(char) * s->nr_cells )) == NULL )
        return error(space_err_malloc);
    bzero( s->cells_taboo , sizeof(char) * s->nr_cells );
    if ( (s->cells_owner = (char *)malloc( sizeof(char) * s->nr_cells )) == NULL )
        return error(space_err_malloc);
    bzero( s->cells_owner , sizeof(char) * s->nr_cells );

    /* allocate the initial partlist */
    if ( ( s->partlist = (struct MxParticle **)malloc( sizeof(struct MxParticle *) * space_partlist_incr ) ) == NULL )
        return error(space_err_malloc);
    if ( ( s->celllist = (struct space_cell **)malloc( sizeof(struct space_cell *) * space_partlist_incr ) ) == NULL )
        return error(space_err_malloc);
    s->nr_parts = 0;
    s->size_parts = space_partlist_incr;

    /* init the cellpair mutexes */
    if ( pthread_mutex_init( &s->tasks_mutex , NULL ) != 0 ||
            pthread_cond_init( &s->tasks_avail , NULL ) != 0 )
        return error(space_err_pthread);

    /* Init the Verlet table (NULL for now). */
    s->verlet_rebuild = 1;
    s->maxdx = 0.0;
    
    // init the large particles cell
    // the large particle cell is at the global origin,
    // so has zero offset for loc.
    l[0] = l[1] = l[1] = 0;
    
    if ( space_cell_init( &s->largeparts, l, s->origin, s->h ) < 0 )
        return error(space_err_cell);

    /* all is well that ends well... */
    return space_err_ok;

}

/**
 * @brief Get the next free #celltuple from the space.
 *
 * @param s The #space in which to look for tuples.
 * @param out A pointer to a #celltuple in which to copy the result.
 * @param wait A boolean value specifying if to wait for free tuples
 *      or not.
 *
 * @return The number of #celltuple found or 0 if the list is empty and
 *      < 0 on error (see #space_err).
 */

int space_gettuple ( struct space *s , struct celltuple **out , int wait ) {

    int i, j, k;
    struct celltuple *t, temp;

    /* Try to get a hold of the cells mutex */
    if ( pthread_mutex_lock( &s->cellpairs_mutex ) != 0 )
        return error(space_err_pthread);

    /* Main loop, while there are still tuples left. */
    while ( s->next_tuple < s->nr_tuples ) {

        /* Loop over all tuples. */
        for ( k = s->next_tuple ; k < s->nr_tuples ; k++ ) {

            /* Put a t on this tuple. */
            t = &( s->tuples[ k ] );

            /* Check if all the cells of this tuple are free. */
            for ( i = 0 ; i < t->n ; i++ )
                if ( s->cells_taboo[ t->cellid[i] ] != 0 )
                    break;
            if ( i < t->n )
                continue;

            /* If so, mark-off the cells pair by pair. */
            for ( i = 0 ; i < t->n ; i++ )
                for ( j = i ; j < t->n ; j++ )
                    if ( t->pairid[ space_pairind(i,j) ] >= 0 ) {
                        s->cells_taboo[ t->cellid[i] ] += 1;
                        s->cells_taboo[ t->cellid[j] ] += 1;
                    }

            /* Swap this tuple to the top of the list. */
            if ( k != s->next_tuple ) {
                temp = s->tuples[k];
                s->tuples[k] = s->tuples[ s->next_tuple ];
                s->tuples[ s->next_tuple ] = temp;
                s->nr_swaps += 1;
            }

            /* Copy this tuple out. */
            *out = &( s->tuples[ s->next_tuple ] );

            /* Increase the top of the list. */
            s->next_tuple += 1;

            /* If this was the last tuple, broadcast to all waiting
                runners to go home. */
            if ( s->next_tuple == s->nr_tuples )
                if (pthread_cond_broadcast(&s->cellpairs_avail) != 0)
                    return error(space_err_pthread);

            /* And leave. */
            if ( pthread_mutex_unlock( &s->cellpairs_mutex ) != 0 )
                return error(space_err_pthread);
            return 1;

        }

        /* If we got here without catching anything, wait for a sign. */
        if ( wait ) {
            s->nr_stalls += 1;
            if ( pthread_cond_wait( &s->cellpairs_avail , &s->cellpairs_mutex ) != 0 )
                return error(space_err_pthread);
        }
        else
            break;

    }

    /* Release the cells mutex */
    if ( pthread_mutex_unlock( &s->cellpairs_mutex ) != 0 )
        return error(space_err_pthread);

    /* Bring good tidings. */
    return space_err_ok;

}


/**
 * @brief Get the next unprocessed cell from the spaece.
 *
 * @param s The #space.
 * @param out Pointer to a pointer to #cell in which to store the results.
 *
 * @return @c 1 if a cell was found, #space_err_ok if the list is empty
 *      or < 0 on error (see #space_err).
 */

int space_getcell ( struct space *s , struct space_cell **out ) {

    int res = 0;

    /* Are there any cells left? */
    if ( s->next_cell == s->nr_cells )
        return 0;

    /* Try to get a hold of the cells mutex */
    if ( pthread_mutex_lock( &s->cellpairs_mutex ) != 0 )
        return error(space_err_pthread);

    /* Try to get a cell. */
    if ( s->next_cell < s->nr_cells ) {
        *out = &( s->cells[ s->next_cell ] );
        s->next_cell += 1;
        res = 1;
    }

    /* Release the cells mutex */
    if ( pthread_mutex_unlock( &s->cellpairs_mutex ) != 0 )
        return error(space_err_pthread);

    /* We've got it! */
    return res;

}


/**
 * @brief Collect forces and potential energies
 *
 * @param s The #space.
 * @param maxcount The maximum number of entries.
 * @param from Pointer to an integer which will contain the index to the
 *        first entry on success.
 * @param to Pointer to an integer which will contain the index to the
 *        last entry on success.
 *
 * @return The number of entries returned or < 0 on error (see #space_err).
 */

int space_verlet_force ( struct space *s , FPTYPE *f , double epot ) {

    int cid, pid, k, ind;
    struct space_cell *c;
    struct MxParticle *p;
    int nr_cells = s->nr_cells, *scells;

    /* Allocate a buffer to mix-up the cells. */
    if ( ( scells = (int *)alloca( sizeof(int) * nr_cells ) ) == NULL )
        return error(space_err_malloc);

    /* Mix-up the order of the cells. */
    for ( k = 0 ; k < nr_cells ; k++ )
        scells[k] = k;
    for ( k = 0 ; k < nr_cells ; k++ ) {
        cid = rand() % nr_cells;
        pid = scells[k]; scells[k] = scells[cid]; scells[cid] = pid;
    }

    /* Loop over the cells. */
    for ( cid = 0 ; cid < nr_cells ; cid++ ) {

        /* Get a pointer on the cell. */
        c = &(s->cells[scells[cid]]);

        /* Get a lock on the cell. */
        if ( pthread_mutex_lock( &c->cell_mutex ) != 0 )
            return error(space_err_pthread);

        for ( pid = 0 ; pid < c->count ; pid++ ) {
            p = &(c->parts[pid]);
            ind = 4 * p->id;
            for ( k = 0 ; k < 3 ; k++ )
                p->f[k] += f[ ind + k ];
        }

        /* Release the cells mutex */
        if ( pthread_mutex_unlock( &c->cell_mutex ) != 0 )
            return error(space_err_pthread);

    }

    /* Add the potential energy to the space's potential energy. */
    if ( pthread_mutex_lock( &s->verlet_force_mutex ) != 0 )
        return error(space_err_pthread);
    s->epot += epot;
    if ( pthread_mutex_unlock( &s->verlet_force_mutex ) != 0 )
        return error(space_err_pthread);

    /* Relax. */
    return space_err_ok;

}


/**
 * @brief Free the cells involved in the current pair.
 *
 * @param s The #space to operate on.
 * @param ci ID of the first cell.
 * @param cj ID of the second cell.
 *
 * @returns #space_err_ok or < 0 on error (see #space_err).
 *
 * Decreases the taboo-counter of the cells involved in the pair
 * and signals any #runner that might be waiting.
 * Note that only a single waiting #runner is released per released cell
 * and therefore, if two different cells become free, the condition
 * @c cellpairs_avail is signaled twice.
 */

int space_releasepair ( struct space *s , int ci , int cj ) {

    /* release the cells in the given pair */
    if ( --(s->cells_taboo[ ci ]) == 0 )
        if (pthread_cond_signal(&s->cellpairs_avail) != 0)
            return error(space_err_pthread);
    if ( --(s->cells_taboo[ cj ]) == 0 )
        if (pthread_cond_signal(&s->cellpairs_avail) != 0)
            return error(space_err_pthread);

    /* all is well... */
    return space_err_ok;

}

/**
 * @brief Initialize the Verlet-list data structures.
 *
 * @param s The #space.
 *
 * @return #space_err_ok or < 0 on error (see #space_err).
 */

int space_verlet_init ( struct space *s , int list_global ) {

    /* Check input for nonsense. */
    if ( s == NULL )
        return error(space_err_null);

    /* Allocate the parts and nrpairs lists if necessary. */
    if ( list_global && s->verlet_size < s->nr_parts ) {

        printf("space_verlet_init: (re)allocating verlet lists...\n");

        /* Free old lists if necessary. */
        if ( s->verlet_list != NULL )
            free( s->verlet_list );
        if ( s->verlet_nrpairs != NULL )
            free( s->verlet_nrpairs );

        /* Allocate new arrays. */
        s->verlet_size = 1.1 * s->nr_parts;
        if ( ( s->verlet_list = (struct verlet_entry *)malloc( sizeof(struct verlet_entry) * s->verlet_size * space_verlet_maxpairs ) ) == NULL )
            return error(space_err_malloc);
        if ( ( s->verlet_nrpairs = (int *)malloc( sizeof(int) * s->verlet_size ) ) == NULL )
            return error(space_err_malloc);

        /* We have to re-build the list now. */
        s->verlet_rebuild = 1;

    }

    /* re-set the Verlet list index. */
    s->verlet_next = 0;

    /* All done! */
    return space_err_ok;

}


/**
 * @brief Generate the list of #celltuple.
 *
 * @param s Pointer to the #space to make tuples for.
 *
 * @return #space_err_ok or < 0 on error (see #space_err).
 */

int space_maketuples ( struct space *s ) {

    int size, incr, *w, w_max, iw_max;
    int i, j, k, kk, pid;
    int ppc, *c2p, *c2p_count;
    struct celltuple *t;
    struct cellpair *p, *p2;

    /* Check for bad input. */
    if ( s == NULL )
        return error(space_err_null);

    /* Clean up any old tuple data that may be lying around. */
    if ( s->tuples != NULL )
        free( s->tuples );

    /* Guess the size of the tuple array and allocate it. */
    size = 1.2 * s->nr_pairs / space_maxtuples;
    if ( ( s->tuples = (struct celltuple *)malloc( sizeof(struct celltuple) * size ) ) == NULL )
        return error(space_err_malloc);
    bzero( s->tuples , sizeof(struct celltuple) * size );
    s->nr_tuples = 0;

    /* Allocate the vector w. */
    if ( ( w = (int *)alloca( sizeof(int) * s->nr_cells ) ) == NULL )
        return error(space_err_malloc);
    s->next_pair = 0;

    /* Allocate and fill the cell-to-pair array. */
    ppc = ( 2*ceil( s->cutoff * s->ih[0] ) + 1 ) * ( 2*ceil( s->cutoff * s->ih[1] ) + 1 ) * ( 2*ceil( s->cutoff * s->ih[2] ) + 1 );
    if ( ( c2p = (int *)alloca( sizeof(int) * s->nr_cells * ppc ) ) == NULL ||
            ( c2p_count = (int *)alloca( sizeof(int) * s->nr_cells ) ) == NULL )
        return error(space_err_malloc);
    bzero( c2p_count , sizeof(int) * s->nr_cells );
    for ( k = 0 ; k < s->nr_pairs ; k++ ) {
        i = s->pairs[k].i; j = s->pairs[k].j;
        c2p[ i*ppc + c2p_count[i] ] = k;
        c2p_count[i] += 1;
        if ( i != j ) {
            c2p[ j*ppc + c2p_count[j] ] = k;
            c2p_count[j] += 1;
        }
    }

    /* While there are still pairs that are not part of a tuple... */
    while ( 1 ) {

        /* Is the array of tuples long enough? */
        if ( s->nr_tuples >= size ) {
            incr = size * 0.2;
            if ( ( t = (struct celltuple *)malloc( sizeof(struct celltuple) * (size + incr) ) ) == NULL )
                return error(space_err_malloc);
            memcpy( t , s->tuples , sizeof(struct celltuple) * size );
            bzero( &t[size] , sizeof(struct celltuple) * incr );
            size += incr;
            free( s->tuples );
            s->tuples = t;
        }

        /* Look for a cell that has free pairs. */
        for ( i = 0 ; i < s->nr_cells && c2p_count[i] == 0 ; i++ );
        if ( i == s->nr_cells )
            break;
        pid = c2p[ i*ppc ];
        p = &( s->pairs[ pid ] );

        /* Get a pointer on the next free tuple. */
        t = &( s->tuples[ s->nr_tuples++ ] );

        /* Clear the t->pairid. */
        for ( k = 0 ; k < space_maxtuples * (space_maxtuples + 1) / 2 ; k++ )
            t->pairid[k] = -1;

        /* Just put the next pair into this tuple. */
        t->cellid[0] = p->i; t->n = 1;
        if ( p->j != p->i ) {
            t->cellid[ t->n++ ] = p->j;
            t->pairid[ space_pairind(0,1) ] = pid;
        }
        else
            t->pairid[ space_pairind(0,0) ] = pid;
        /* printf("space_maketuples: starting tuple %i with pair [%i,%i].\n",
            s->nr_tuples-1 , p->i , p->j ); */

        /* Remove this pair from the c2ps. */
        for ( k = 0 ; k < c2p_count[p->i] ; k++ )
            if ( c2p[ p->i*ppc + k ] == pid ) {
                c2p_count[p->i] -= 1;
                c2p[ p->i*ppc + k ] = c2p[ p->i*ppc + c2p_count[p->i] ];
                break;
            }
        if ( p->i != p->j )
            for ( k = 0 ; k < c2p_count[p->j] ; k++ )
                if ( c2p[ p->j*ppc + k ] == pid ) {
                    c2p_count[p->j] -= 1;
                    c2p[ p->j*ppc + k ] = c2p[ p->j*ppc + c2p_count[p->j] ];
                    break;
                }

        /* Add self-interactions, if any. */
        if ( p->i != p->j ) {
            for ( k = 0 ; k < c2p_count[p->i] ; k++ ) {
                p2 = &( s->pairs[ c2p[ p->i*ppc + k ] ] );
                if ( p2->i == p2->j ) {
                    t->pairid[ space_pairind(0,0) ] = c2p[ p->i*ppc + k ];
                    c2p_count[p->i] -= 1;
                    c2p[ p->i*ppc + k ] = c2p[ p->i*ppc + c2p_count[p->i] ];
                    break;
                }
            }
            for ( k = 0 ; k < c2p_count[p->j] ; k++ ) {
                p2 = &( s->pairs[ c2p[ p->j*ppc + k ] ] );
                if ( p2->i == p2->j ) {
                    t->pairid[ space_pairind(1,1) ] = c2p[ p->j*ppc + k ];
                    c2p_count[p->j] -= 1;
                    c2p[ p->j*ppc + k ] = c2p[ p->j*ppc + c2p_count[p->j] ];
                    break;
                }
            }
        }

        /* Fill the weights for the cells. */
        bzero( w , sizeof(int) * s->nr_cells );
        for ( k = 0 ; k < t->n ; k++ )
            w[ t->cellid[k] ] = -1;
        for ( i = 0 ; i < t->n ; i++ ) {
            for ( k = 0 ; k < c2p_count[ t->cellid[i] ] ; k++ ) {
                p = &( s->pairs[ c2p[ t->cellid[i]*ppc + k ] ] );
                if ( p->i == t->cellid[i] && w[ p->j ] >= 0 )
                    w[ p->j ] += 1;
                if ( p->j == t->cellid[i] && w[ p->i ] >= 0 )
                    w[ p->i ] += 1;
            }
        }

        /* Find the cell with the maximum weight. */
        w_max = 0;
        for ( k = 1 ; k < s->nr_cells ; k++ )
            if ( w[k] > w[w_max] )
                w_max = k;

        /* While there is still another cell that can be added... */
        while ( w[w_max] > 0 && t->n < space_maxtuples ) {

            /* printf("space_maketuples: adding cell %i to tuple %i (w[%i]=%i).\n",
                w_max, s->nr_tuples-1, w_max, w[w_max] ); */

            /* Add this cell to the tuple. */
            iw_max = t->n++;
            t->cellid[ iw_max ] = w_max;

            /* Look for pairs that contain w_max and someone from the tuple. */
            k = 0;
            while ( k < c2p_count[w_max] ) {

                /* Get this pair. */
                pid = c2p[ w_max*ppc + k ];
                p = &( s->pairs[ pid ] );

                /* Get the tuple indices of the cells in this pair. */
                if ( p->i == w_max )
                    i = iw_max;
                else
                    for ( i = 0 ; i < t->n && t->cellid[i] != p->i ; i++ );
                if ( p->j == w_max )
                    j = iw_max;
                else
                    for ( j = 0 ; j < t->n && t->cellid[j] != p->j ; j++ );

                /* If this pair is not in the tuple, skip it. */
                if ( i == t->n || j == t->n )
                    k += 1;

                /* Otherwise... */
                else {

                    /* Add this pair to the tuple. */
                    if ( i < j )
                        t->pairid[ space_pairind(i,j) ] = pid;
                    else
                        t->pairid[ space_pairind(j,i) ] = pid;
                    /* printf("space_maketuples: adding pair [%i,%i] to tuple %i (w[%i]=%i).\n",
                        p->i, p->j, s->nr_tuples-1 , w_max , w[w_max] ); */

                    /* Remove this pair from the c2ps. */
                    for ( kk = 0 ; kk < c2p_count[p->i] ; kk++ )
                        if ( c2p[ p->i*ppc + kk ] == pid ) {
                            c2p_count[p->i] -= 1;
                            c2p[ p->i*ppc + kk ] = c2p[ p->i*ppc + c2p_count[p->i] ];
                            break;
                        }
                    if ( p->i != p->j )
                        for ( kk = 0 ; kk < c2p_count[p->j] ; kk++ )
                            if ( c2p[ p->j*ppc + kk ] == pid ) {
                                c2p_count[p->j] -= 1;
                                c2p[ p->j*ppc + kk ] = c2p[ p->j*ppc + c2p_count[p->j] ];
                                break;
                            }
                }

            }

            /* Update the weights and get the ID of the new max. */
            w[ w_max ] = -1;
            for ( k = 0 ; k < c2p_count[w_max] ; k++ ) {
                p = &( s->pairs[ c2p[ w_max*ppc + k ] ] );
                if ( p->i == w_max && w[ p->j ] >= 0 )
                    w[ p->j ] += 1;
                if ( p->j == w_max && w[ p->i ] >= 0 )
                    w[ p->i ] += 1;
            }

            /* Find the cell with the maximum weight. */
            w_max = 0;
            for ( k = 1 ; k < s->nr_cells ; k++ )
                if ( w[k] > w[w_max] )
                    w_max = k;

        }

    }

    /* Dump the list of tuples. */
    /* for ( i = 0 ; i < s->nr_tuples ; i++ ) {
        t = &( s->tuples[i] );
        printf("space_maketuples: tuple %i has pairs:",i);
        for ( k = 0 ; k < t->n ; k++ )
            for ( j = k ; j < t->n ; j++ )
                if ( t->pairid[ space_pairind(k,j) ] >= 0 )
                    printf(" [%i,%i]", t->cellid[j], t->cellid[k] );
        printf("\n");
        } */

    /* If we made it up to here, we're done! */
    return space_err_ok;

}

CAPI_FUNC(HRESULT) space_del_particle(struct space *s, int pid)
{
    if(pid < 0 || pid >= s->size_parts) {
        return c_error(E_FAIL, "pid out of range");
    }
    MxParticle *p = s->partlist[pid];

    if(p == NULL) {
        return c_error(E_FAIL, "particle is already null and deleted");
    }

    space_cell *cell = s->celllist[pid];
    assert(cell && "space cell is null");

    s->partlist[pid] = NULL;
    s->celllist[pid] = NULL;

    // index of cell in cell particle array.
    size_t cid = p - cell->parts;

    assert(p == &cell->parts[cid] && "pointer arithmetic error");
    
    // the particle will get overwritten, release the
    // pyobject ptr now
    Py_DecRef(p->_pyparticle);

    cell->count -= 1;
    if ( cid < cell->count ) {
        cell->parts[cid] = cell->parts[cell->count];
        s->partlist[cell->parts[cid].id ] = &( cell->parts[cid] );
    }

    s->nr_parts -= 1;
    
    MxParticleType *type = &_Engine.types[p->typeId];
    NOMStyle *style = p->style ? p->style : type->style;
    
    if(style->flags & STYLE_VISIBLE) {
        if(p->flags & PARTICLE_LARGE) {
            s->nr_visable_large_parts -= 1;
        }
        else {
            s->nr_visable_parts -= 1;
        }
    }

    return S_OK;
}


int space_get_cellids_for_pos (struct space *s , FPTYPE *x, int *cellids) {
    
    int k, ind[3];

    /* get the hypothetical cell coordinate */
    for ( k = 0 ; k < 3 ; k++ ) {
        ind[k] = (x[k] - s->origin[k]) * s->ih[k];
    }
    
    if(cellids) {
        for ( k = 0 ; k < 3 ; k++ ) {
            cellids[k] = ind[k];
        }
    }
    
    /* is this particle within the space? */
    for ( k = 0 ; k < 3 ; k++ )
        if ( ind[k] < 0 || ind[k] >= s->cdim[k] )
            return error(space_err_range);
    
    return space_cellid(s,ind[0],ind[1],ind[2]);
}


HRESULT space_update_style( struct space *s ) {
    s->nr_visable_parts = 0;
    s->nr_visable_large_parts = 0;
    
    for(int i = 0; i < s->nr_parts; ++i) {
        MxParticle *p = s->partlist[i];
        MxParticleType *type = &_Engine.types[p->typeId];
        NOMStyle *style = p->style ? p->style : type->style;
        
        if(style->flags & STYLE_VISIBLE) {
            if(p->flags & PARTICLE_LARGE) {
                s->nr_visable_large_parts +=1;
            }
            else {
                s->nr_visable_parts += 1;
            }
        }
    }
    return S_OK;
}
