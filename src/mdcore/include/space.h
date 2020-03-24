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
#ifndef INCLUDE_SPACE_H_
#define INCLUDE_SPACE_H_
#include "mdcore_config.h"


MDCORE_BEGIN_DECLS

/* space error codes */
#define space_err_ok                    0
#define space_err_null                  -1
#define space_err_malloc                -2
#define space_err_cell                  -3
#define space_err_pthread               -4
#define space_err_range                 -5
#define space_err_maxpairs              -6
#define space_err_nrtasks               -7
#define space_err_task                  -8


/* some constants */
#define space_periodic_none             0
#define space_periodic_x                1
#define space_periodic_y                2
#define space_periodic_z                4
#define space_periodic_full             7
#define space_periodic_ghost_x          8
#define space_periodic_ghost_y          16
#define space_periodic_ghost_z          32
#define space_periodic_ghost_full       56

#define space_partlist_incr             100

/** Maximum number of cells per tuple. */
#define space_maxtuples                 4

/** Maximum number of interactions per particle in the Verlet list. */
#define space_verlet_maxpairs           800


/* some useful macros */
/** Converts the index triplet (@c i, @c j, @c k) to the cell id in the
    #space @c s. */
#define space_cellid(s,i,j,k)           (  ((i)*(s)->cdim[1] + (j)) * (s)->cdim[2] + (k) )

/** Convert tuple ids into the pairid index. */
#define space_pairind(i,j)              ( space_maxtuples*(i) - (i)*((i)+1)/2 + (j) )

/** ID of the last error */
extern int space_err;

/** Struct for Verlet list entries. */
struct verlet_entry {

    /** The particle. */
    struct MxParticle *p;

    /** The interaction potential. */
    struct MxPotential *pot;

    /** The integer shift relative to this particle. */
    signed char shift[3];

};

/** Struct for each cellpair (see #space_getpair). */
struct cellpair {

    /** Indices of the cells involved. */
    int i, j;

    /** Relative shift between cell centres. */
    FPTYPE shift[3];

    /** Pairwise Verlet stuff. */
    int size, count;
    short int *pairs;
    short int *nr_pairs;

    /** Pointer to chain pairs together. */
    struct cellpair *next;

};

/** Struct for groups of cellpairs. */
struct celltuple {

    /** IDs of the cells in this tuple. */
    int cellid[ space_maxtuples ];

    /** Nr. of cells in this tuple. */
    int n;

    /** Ids of the underlying pairs. */
    int pairid[ space_maxtuples * (space_maxtuples + 1) / 2 ];

    /* Buffer value to keep the size at 64 bytes for space_maxtuples==4. */
    int buff;

};


/**
 * The space structure
 */
typedef struct space {
    PyObject_HEAD

    /** Real dimensions. */
    double dim[3];

    /** Location of origin. */
    double origin[3];

    /** Space dimension in cells. */
    int cdim[3];

    /** Number of cells within cutoff in each dimension. */
    int span[3];

    /** Cell edge lengths and their inverse. */
    double h[3], ih[3];

    /** The cutoff and the cutoff squared. */
    double cutoff, cutoff2;

    /** Periodicities. */
    unsigned int period;

    /** Total nr of cells in this space. */
    int nr_cells;

    /** IDs of real, ghost and marked cells. */
    int *cid_real, *cid_ghost, *cid_marked;
    int nr_real, nr_ghost, nr_marked;

    /** Array of cells spanning the space. */
    struct space_cell *cells;

    /** The total number of tasks. */
    int nr_tasks, tasks_size;

    /** Array of tasks. */
    struct task *tasks;

    /** Condition/mutex to signal task availability. */
    pthread_mutex_t tasks_mutex;
    pthread_cond_t tasks_avail;

    /** Taboo-list for collision avoidance */
    char *cells_taboo;

    /** Id of #runner owning each cell. */
    char *cells_owner;

    /** Counter for the number of swaps in every step. */
    int nr_swaps, nr_stalls;

    /** Array of pointers to the individual parts, sorted by their ID. */
    struct MxParticle **partlist;

    /** Array of pointers to the #cell of individual parts, sorted by their ID. */
    struct space_cell **celllist;

    /** Number of parts in this space and size of the buffers partlist and celllist. */
    int nr_parts, size_parts;

    /** Trigger re-building the cells/sorts. */
    int verlet_rebuild;

    /** The maximum particle displacement over all cells. */
    FPTYPE maxdx;

    /** Potential energy collected by the space itself. */
    double epot, epot_nonbond, epot_bond, epot_angle, epot_dihedral, epot_exclusion;

    /** Data for the verlet list. */
    struct verlet_entry *verlet_list;
    FPTYPE *verlet_oldx, verlet_maxdx;
    int *verlet_nrpairs;
    int verlet_size, verlet_next;
    pthread_mutex_t verlet_force_mutex;





    /** The total number of cell pairs. */
    int nr_pairs;

    /** Array of cell pairs. */
    struct cellpair *pairs;

    /** Id of the next unprocessed pair (for #space_getpair) */
    int next_pair;

    /** Array of cell tuples. */
    struct celltuple *tuples;

    /** The number of tuples. */
    int nr_tuples;

    /** The ID of the next free tuple or cell. */
    int next_tuple, next_cell;

    /** Mutex for accessing the cell pairs. */
    pthread_mutex_t cellpairs_mutex;

    /** Spin-lock for accessing the cell pairs. */
    //lock_type cellpairs_spinlock;

    /** Condition to wait for free cells on. */
    pthread_cond_t cellpairs_avail;

} space;


/* associated functions */
int space_init ( struct space *s , const double *origin , const double *dim , double *L , double cutoff , unsigned int period );
int space_getsid ( struct space *s , struct space_cell **ci , struct space_cell **cj , FPTYPE *shift );
int space_shuffle ( struct space *s );
int space_shuffle_local ( struct space *s );
int space_addpart ( struct space *s , struct MxParticle *p , double *x );
int space_prepare ( struct space *s );
int space_getpos ( struct space *s , int id , double *x );
int space_setpos ( struct space *s , int id , double *x );
int space_flush ( struct space *s );
int space_flush_ghosts ( struct space *s );
struct task *space_addtask ( struct space *s , int type , int subtype , int flags , int i , int j );


int space_verlet_init ( struct space *s , int list_global );
int space_gettuple ( struct space *s , struct celltuple **out , int wait );
int space_getcell ( struct space *s , struct space_cell **out );
int space_verlet_force ( struct space *s , FPTYPE *f , double epot );
int space_releasepair ( struct space *s , int ci , int cj );

MDCORE_END_DECLS
#endif // INCLUDE_SPACE_H_
