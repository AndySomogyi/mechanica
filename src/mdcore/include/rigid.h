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
#ifndef INCLUDE_RIGID_H_
#define INCLUDE_RIGID_H_
#include "platform.h"

MDCORE_BEGIN_DECLS

/* rigid error codes */
#define rigid_err_ok                    0
#define rigid_err_null                  -1
#define rigid_err_malloc                -2


/* Some constants. */
#define rigid_maxparts                  10
#define rigid_maxconstr                 (3*rigid_maxparts)
#define rigid_maxiter                   100
#define rigid_pshake_refine             4
#define rigid_pshake_maxalpha           0.1f


/** ID of the last error */
extern int rigid_err;


/** The rigid structure */
typedef struct rigid {

	/** Nr of parts involved. */
	int nr_parts;

	/** ids of particles involved */
	int parts[ rigid_maxparts ];

	/** Nr of constraints involved. */
	int nr_constr;

	/** The constraints themselves. */
	struct {
		int i, j;
		double d2;
	} constr[ rigid_maxconstr ];

	/** The constraint shuffle matrix. */
	float a[ rigid_maxconstr*rigid_maxconstr ];

} rigid;


/* associated functions */
int rigid_eval_shake ( struct rigid *r , int N , struct engine *e );
int rigid_eval_pshake ( struct rigid *r , int N , struct engine *e , int a_update );

MDCORE_END_DECLS
#endif // INCLUDE_RIGID_H_
