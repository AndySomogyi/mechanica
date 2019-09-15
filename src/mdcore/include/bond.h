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

#ifndef INCLUDE_BOND_H_
#define INCLUDE_BOND_H_
#include "platform.h"

/* bond error codes */
#define bond_err_ok                    0
#define bond_err_null                  -1
#define bond_err_malloc                -2


/** ID of the last error */
extern int bond_err;


/** The bond structure */
typedef struct bond {

	/* ids of particles involved */
	int i, j;

} bond;


MDCORE_BEGIN_DECLS

/* associated functions */
int bond_eval ( struct bond *b , int N , struct engine *e , double *epot_out );
int bond_evalf ( struct bond *b , int N , struct engine *e , FPTYPE *f , double *epot_out );

MDCORE_END_DECLS

#endif // INCLUDE_BOND_H_
