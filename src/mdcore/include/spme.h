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
#ifndef INCLUDE_SPME_H_
#define INCLUDE_SPME_H_
#include "platform.h"

MDCORE_BEGIN_DECLS

/* spme error codes */
#define spme_err_ok                    0
#define spme_err_null                  -1
#define spme_err_malloc                -2
#define smpe_err_nofftw3               -3
#define spme_err_fftw                  -4


/* some constants */
#define spme_order                     4
#define spme_gpc                       4


/** ID of the last error */
extern int spme_err;


/** The spme structure */
typedef struct spme {

#ifdef HAVE_FFTW3

	/** Grid dimensions. */
	int dim[3];

	/** Grid spacing. */
	float h[3], ih[3];

	/** SMPE parameter. */
	float kappa;

	/** The charge grid. */
	fftwf_complex *Q;

	/** The transformed grid. */
	fftwf_complex *E;

	/** The Temporary, complex grid. */
	fftwf_complex *T;

	/** The structure array. */
	float *theta;

	/** The fftw plan. */
	fftwf_plan fwplan, bwplan;

#endif

} spme;


/* associated functions */
int spme_init ( struct spme *s , int *dim , float *h , float kappa );
void spme_bspline ( float *x , int N , float *b , float *dbdx );
int spme_doconv ( struct spme *s );
void spme_iact ( struct spme *s , struct space_cell *cp , struct space_cell *cg );

MDCORE_END_DECLS
#endif // INCLUDE_SPME_H_
