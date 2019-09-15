/*******************************************************************************
 * This file is part of mdcore.
 * Coypright (c) 2013 Pedro Gonnet (pedro.gonnet@durham.ac.uk)
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

/* Include configuration header */
#include "config.h"

/* Include some standard header files */
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <limits.h>

/* Include some conditional headers. */
#include "config.h"
#ifdef WITH_MPI
    #include <mpi.h>
#endif

/* Include local headers */
#include "cycle.h"
#include "errs.h"
#include "fptype.h"
#include "lock.h"
#include "task.h"


/* Global variables. */
/** The ID of the last error. */
int task_err = task_err_ok;
unsigned int task_rcount = 0;

/* the error macro. */
#define error(id)				( task_err = errs_register( id , task_err_msg[-(id)] , __LINE__ , __FUNCTION__ , __FILE__ ) )

/* list of error messages. */
char *task_err_msg[4] = {
	"Nothing bad happened.",
    "An unexpected NULL pointer was encountered.",
    "A call to malloc failed, probably due to insufficient memory.",
    "Attempted to add an unlock to a full task.",
	};
    
    

/**
 * @brief Add a task dependency.
 * 
 * @param ta The unlocking #task.
 * @param tb The unlocked #task.
 *
 */
 
int task_addunlock ( struct task *ta , struct task *tb ) {

    /* Is there space for this? */
    if ( ta->nr_unlock >= task_max_unlock )
        return error(task_err_maxunlock);

    /* Add the unlock. */
    ta->unlock[ ta->nr_unlock ] = tb;
    ta->nr_unlock += 1;
    
    /* Ta-da! */
    return task_err_ok;
    
    }

    
