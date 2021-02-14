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

#pragma once
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

enum CellFlags {
    cell_flag_none    = 0,
    cell_flag_ghost   = 1 << 0,
    cell_flag_wait    = 1 << 1,
    cell_flag_waited  = 1 << 2,
    cell_flag_marked  = 1 << 3,
    cell_flag_large   = 1 << 4,
    
    cell_boundary_top      = 1 << 5,
    cell_boundary_bottom   = 1 << 6,
    cell_boundary_left     = 1 << 7,
    cell_boundary_right    = 1 << 8,
    cell_boundary_front    = 1 << 9,
    cell_boundary_back     = 1 << 10,
    cell_boundary_any      = cell_boundary_top |
                             cell_boundary_bottom |
                             cell_boundary_left |
                             cell_boundary_right |
                             cell_boundary_front |
                             cell_boundary_back
};





/* Map shift vector to sortlist. */
CAPI_DATA(const char) cell_sortlistID[27];
CAPI_DATA(const FPTYPE) cell_shift[13*3];
CAPI_DATA(const char) cell_flip[27];


/* the last error */
CAPI_DATA(int) cell_err;


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
	struct MxParticle *parts;

	/* buffer to store the potential energy */
	double epot;

	/* a buffer to store incomming parts. */
	struct MxParticle *incomming;
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

/**
 * @brief Initialize the given cell.
 *
 * @param c The #cell to initialize.
 * @param loc Array containing the location of this cell in the space.
 * @param origin The origin of the cell in global coordinates
 * @param dim The cell dimensions.
 *
 * @return #cell_err_ok or < 0 on error (see #cell_err).
 */
int space_cell_init ( struct space_cell *c , int *loc , double *origin ,
        double *dim );

/**
 * @brief Add a particle to a cell.
 *
 * @param c The #cell to which the particle should be added.
 * @param p The #particle to add to the cell
 *
 * @return A pointer to the particle data in the cell.
 *
 * This routine assumes the particle position has already been adjusted
 * to the cell @c c.
 */
struct MxParticle *space_cell_add ( struct space_cell *c ,
        struct MxParticle *p , struct MxParticle **partlist );

/**
 * @brief Add a particle to the incomming array of a cell.
 *
 * @param c The #cell to which the particle should be added.
 * @param p The #particle to add to the cell
 *
 * @return A pointer to the particle data in the incomming array of
 *      the cell.
 *
 * This routine assumes the particle position has already been adjusted
 * to the cell @c c.
 */
struct MxParticle *space_cell_add_incomming ( struct space_cell *c ,
        struct MxParticle *p );

/**
 * @brief Add one or more particles to the incomming array of a cell.
 *
 * @param c The #cell to which the particle should be added.
 * @param p The #particle to add to the cell
 *
 * @return The number of incomming parts or < 0 on error (see #cell_err).
 *
 * This routine assumes the particle position have already been adjusted
 * to the cell @c c.
 */
int space_cell_add_incomming_multiple ( struct space_cell *c ,
        struct MxParticle *p , int count );

/**
 * @brief Move particles from the incomming buffer to the cell.
 *
 * @param c The #cell.
 * @param partlist A pointer to the partlist to set the part indices.
 *
 * @return #cell_err_ok or < 0 on error (see #cell_err).
 */
int space_cell_welcome ( struct space_cell *c ,
        struct MxParticle **partlist );

/**
 * @brief Load a block of particles to the cell.
 *
 * @param c The #cell.
 * @param parts Pointer to a block of #part.
 * @param nr_parts The number of parts to load.
 * @param partlist A pointer to the partlist to set the part indices.
 * @param celllist A pointer to the celllist to set the part indices.
 *
 * @return #cell_err_ok or < 0 on error (see #cell_err).
 */
int space_cell_load ( struct space_cell *c , struct MxParticle *parts ,
        int nr_parts , struct MxParticle **partlist , struct space_cell **celllist );

/**
 * @brief Flush all the parts from a #cell.
 *
 * @param c The #cell to flush.
 * @param partlist A pointer to the partlist to set the part indices.
 * @param celllist A pointer to the celllist to set the part indices.
 *
 * @return #cell_err_ok or < 0 on error (see #cell_err).
 */
int space_cell_flush ( struct space_cell *c ,
        struct MxParticle **partlist , struct space_cell **celllist );

#include <iostream>
std::ostream& operator<<(std::ostream& os, const space_cell* );




#endif // INCLUDE_SPACE_CELL_H_
