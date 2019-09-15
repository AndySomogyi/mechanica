/*******************************************************************************
 * This file is part of mdcore.
 * Coypright (c) 2013 Pedro Gonnet (pedro.gonnet@durham.ac.uk)
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
#ifndef INCLUDE_TASK_H_
#define INCLUDE_TASK_H_
#include "platform.h"

MDCORE_BEGIN_DECLS

/* task error codes */
#define task_err_ok                    0
#define task_err_null                  -1
#define task_err_malloc                -2
#define task_err_maxunlock             -3


/* some constants */
#define task_max_unlock                124


/** Task types. */
enum {
	task_type_none = 0,
	task_type_self,
	task_type_pair,
	task_type_sort,
	task_type_bonded,
	task_type_count
};


/** Task subtypes. */
enum {
	task_subtype_none = 0,
	task_subtype_real,
	task_subtype_spme,
	task_subtype_count
};


/** ID of the last error */
extern int task_err;


/** The task structure */
typedef struct task {

	/** Task type/subtype. */
	short int type, subtype;

	/** Wait counters. */
	int wait;

	/** Task flags. */
	int flags;

	/** Indices of the cells/domain involved. */
	int i, j;

	/** Nr of task that this task unlocks. */
	int nr_unlock;

	/** List of task that this task unlocks (dependencies). */
	struct task *unlock[ task_max_unlock ];

} task;


/* associated functions */
int task_addunlock ( struct task *ta , struct task *tb );

MDCORE_END_DECLS
#endif // INCLUDE_TASK_H_

