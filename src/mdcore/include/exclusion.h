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
#ifndef INCLUDE_EXCLUSION_H_
#define INCLUDE_EXCLUSION_H_
#include "platform.h"

MDCORE_BEGIN_DECLS

/* exclusion error codes */
#define exclusion_err_ok                    0
#define exclusion_err_null                  -1
#define exclusion_err_malloc                -2


/** ID of the last error */
extern int exclusion_err;


/** The exclusion structure */
typedef struct exclusion {

	/* ids of particles involved */
	int i, j;

} exclusion;


/* associated functions */
int exclusion_eval ( struct exclusion *b , int N , struct engine *e , double *epot_out );
int exclusion_evalf ( struct exclusion *b , int N , struct engine *e , FPTYPE *f , double *epot_out );

MDCORE_END_DECLS
#endif // INCLUDE_EXCLUSION_H_
