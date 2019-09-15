/*******************************************************************************
 * This file is part of mdcore.
 * Coypright (c) 2010 Pedro Gonnet (pedro.gonnet@durham.ac.uk)
 * Coypright (c) 2017 Andy Somogyi (somogyie at indiana dot edu)
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

#ifndef INCLUDE_SPACE_CELL_H_
#define INCLUDE_SPACE_CELL_H_

#include "platform.h"
#include "pthread.h"


/* cell error codes */
#define cell_err_ok                     0
#define cell_err_null                   -1
#define cell_err_malloc                 -2
#define cell_err_pthread                -3


/* some constants */
#define cell_default_size               64
#define cell_incr                       10

/** Alignment when allocating parts. */
#define cell_partalign                  64

/** Cell flags */
#define cell_flag_none                  0
#define cell_flag_ghost                 1
#define cell_flag_wait                  2
#define cell_flag_waited                4
#define cell_flag_marked                8


MDCORE_BEGIN_DECLS


/* Map shift vector to sortlist. */
extern const char cell_sortlistID[27];
extern const FPTYPE cell_shift[13*3];
extern const char cell_flip[27]; 


/* the last error */
extern int cell_err;


/**
 * @brief the space_cell structure
 *
 * The space_cell represents a rectangular region of space, and physically
 * stores all particle data. A set of cells form a uniform rectangular grid.
 *
 * Simulation box divided into cells with size equal to or slightly larger than
 * the largest non-bonded force cutoff distance. Each particle only interacts
 * with others in its own cell or adjacent cells
 */
typedef struct space_cell {

	/* some flags */
	unsigned int flags;

	/* The ID of this cell. */
	int id;

	/* relative cell location */
	int loc[3];

	/* absolute cell origin */
	double origin[3];

	/* cell dimensions */
	double dim[3];

	/* size and count of particle buffer */
	int size, count;

	/* the particle buffer */
	struct particle *parts;

	/* buffer to store the potential energy */
	double epot;

	/* a buffer to store incomming parts. */
	struct particle *incomming;
	int incomming_size, incomming_count;

	/* Mutex for synchronized cell access. */
	pthread_mutex_t cell_mutex;
	pthread_cond_t cell_cond;

	/* Old particle positions for the verlet lists. */
	FPTYPE *oldx;
	int oldx_size;

	/* ID of the node this cell belongs to. */
	int nodeID;

	/* Pointer to sorted cell data. */
	unsigned int *sortlist;

	/* Sorting task for this cell. */
	struct task *sort;

	/*ID of the GPU this cell belongs to. */
	int GPUID;

} space_cell;



/* associated functions */
int space_cell_init ( struct space_cell *c , int *loc , double *origin , double *dim );
struct particle *space_cell_add ( struct space_cell *c , struct particle *p , struct particle **partlist );
struct particle *space_cell_add_incomming ( struct space_cell *c , struct particle *p );
int space_cell_add_incomming_multiple ( struct space_cell *c , struct particle *p , int count );
int space_cell_welcome ( struct space_cell *c , struct particle **partlist );
int space_cell_load ( struct space_cell *c , struct particle *parts , int nr_parts , struct particle **partlist , struct space_cell **celllist );
int space_cell_flush ( struct space_cell *c , struct particle **partlist , struct space_cell **celllist );

MDCORE_END_DECLS

#endif // INCLUDE_SPACE_CELL_H_
