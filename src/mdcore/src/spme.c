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
#include <strings.h>
#include <alloca.h>
#include <pthread.h>
#include <math.h>

/* Include conditional headers. */
#include "config.h"
#ifdef HAVE_OPENMP
    #include <omp.h>
#endif
#ifdef WITH_MPI
    #include <mpi.h>
#endif

/* I will need fftw for this. */
#include <complex.h>
#ifdef HAVE_FFTW3
#include <fftw3.h>
#endif

/* include local headers */
#include "errs.h"
#include "fptype.h"
#include "lock.h"
#include <particle.h>
#include <space_cell.h>
#include "spme.h"


/* the last error */
int spme_err = spme_err_ok;


/* the error macro. */
#define error(id)				( spme_err = errs_register( id , spme_err_msg[-(id)] , __LINE__ , __FUNCTION__ , __FILE__ ) )

/* list of error messages. */
char *spme_err_msg[5] = {
	"Nothing bad happened.",
    "An unexpected NULL pointer was encountered.",
    "A call to malloc failed, probably due to insufficient memory.",
    "SMPE not available, mdcore was not compiled with fftw3.",
    "An error occured when calling an fftw3 funciton."
	};

/* B-spline coeffs. */
float spme_coeffs[12] = { 2.0f/3.0f , 0.0f , -1.0f , 0.5f ,
                          4.0f/3.0f , -2.0f , 1.0f , -1.0f/6.0f ,
                          0.0f , 0.0f , 0.0f , 0.0f };
                          
                        
/**
 * @brief Do the convolution stuff.
 * 
 * @param s The #spme.
 */
 
int spme_doconv ( struct spme *s ) {

#ifdef HAVE_FFTW3
    int k, N = s->dim[0]*s->dim[1]*s->dim[2];
    float scale = 1.0f / N;
    fftwf_complex *T = s->T;

    /* Execute the forward plan. */
    fftwf_execute( s->fwplan );
    
    /* Multiply \hat{Q} with theta. */
    for ( k = 0 ; k < N ; k++ )
        T[k] *= s->theta[k];
        
    /* Reset Q. */
    bzero( s->Q , sizeof(float) * N );
    
    /* Execute the backward plan. */
    fftwf_execute( s->bwplan );
    
    /* Scale the result back. */
    for ( k = 0 ; k < N ; k++ )
        s->E[k] *= scale;

    return spme_err_ok;
    
#else
    return error(smpe_err_nofftw3);
#endif

    }


/**
 * @brief Compute the interactions between the particles and grid
 *
 * @param spme The #spme object.
 * @param cp The #cell containing the parts
 * @param cg The #cell containing the grid
 * @param shift The shift vector from @c cp to @c cg.
 *
 */
 
void spme_iact ( struct spme *restrict s , struct space_cell *restrict cp , struct space_cell *restrict cg ) {

#ifdef HAVE_FFTW3
    int j, k, pid, ind[3], dim[3];
    struct part *restrict p, *restrict parts = cp->parts;
    FPTYPE shift[3], px[3], ih[3];
    float q;
    fftwf_complex *Q, *E;
    float x[ 3*spme_gpc ], b[ 3*spme_gpc ], dbdx[ 3*spme_gpc ];
    int minx[3], maxx[3], off[3], count;
    
    /* Compute the pshift. */
    for ( k = 0 ; k < 3 ; k++ ) {
        shift[k] = cp->origin[k] - cg->origin[k];
        if ( shift[k] * 2 > s->dim[k] )
            shift[k] -= s->dim[k];
        else if ( shift[k] * 2 < -s->dim[k] )
            shift[k] += s->dim[k];
        }

    /* Get the origin of the grid in cg. */
    for ( k = 0 ; k < 3 ; k++ ) {
        dim[k] = s->dim[k];
        ih[k] = s->ih[k];
        ind[k] = cg->loc[k] * spme_gpc;
        }
    Q = &s->Q[ ind[2] + dim[2]*( ind[1] + dim[1]*ind[0] ) ];
    E = &s->E[ ind[2] + dim[2]*( ind[1] + dim[1]*ind[0] ) ];

    /* Loop over the parts in cp. */
    for ( pid = 0 ; pid < cp->count ; pid++ ) {
    
        /* Get the particle. */
        p = &parts[pid];
        
        /* Does this particle even have a charge? */
        q = p->q;
        if ( q == 0.0f )
            continue;
        
        /* Get the location (normalized) */
        for ( k = 0 ; k < 3 ; k++ ) {
            px[k] = ( p->x[k] + shift[k] ) * ih[k];
            minx[k] = px[k] - spme_gpc/2 + 0.5f;
            maxx[k] = minx[k] + 3;
            minx[k] = ( minx[k] < 0 ) ? 0 : minx[k];
            maxx[k] = ( maxx[k] > spme_gpc-1 ) ? spme_gpc-1 : maxx[k];
            }
            
        /* Fill the distance vectors in each dimension. */
        for ( count = 0 , k = 0 ; k < 3 ; k++ ) {
            off[k] = count;
            for ( j = minx[k] ; j <= maxx[k] ; j++ ) {
                x[count] = fabsf( px[k] - j );
                count += 1;
                }
            }
            
        if ( count == 0 )
            continue;
                
        /* Evaluate the b-spline components. */
        spme_bspline( x , count , b , dbdx );
            
        /* Loop over the grid points in cg. */
        for ( ind[0] = minx[0] ; ind[0] <= maxx[0] ; ind[0]++ )
            for ( ind[1] = minx[1] ; ind[1] <= maxx[1] ; ind[1]++ )
                for ( ind[2] = minx[2] ; ind[2] <= maxx[2] ; ind[2]++ ) {
                
                    /* Apply the interaction to the grid. */
                    Q[ ind[2] + dim[2]*( ind[1] + dim[1]*ind[0] ) ] += q * b[ off[0] + ind[0]-minx[0] ] * b[ off[1] + ind[1]-minx[1] ] * b[ off[2] + ind[2]-minx[2] ];
                    
                    /* Apply the interaction to the particle. */
                    for ( k = 0 ; k < 3 ; k++ )
                        p->f[k] += q * E[ ind[2] + dim[2]*( ind[1] + dim[1]*ind[0] ) ] * dbdx[ off[k] + ind[k]-minx[k] ] * ih[k];
                
                    } /* loop over grid cells. */
    
        } /* loop over parts in cp. */

#endif

    }


/**
 * @brief Evaluate the B-spline and its derivatives at the given points
 *
 * @param x The distances, normalized.
 * @param N The number of points.
 * @param b A vector in which to store the B-splines.
 * @param dbdx A vector in which to store the derivative.
 *
 */
 
void spme_bspline ( float *x , int N , float *b , float *dbdx ) {

    int i, k, ival[N];
    
    /* Get the interval for each component. */
    for ( k = 0 ; k < N ; k++ )
        ival[k] = fminf( x[k] , 2.0f );
        
    /* Init the horner scheme. */
    for ( k = 0 ; k < N ; k++ ) {
        b[k] = spme_coeffs[ 4*ival[k] + 3 ];
        dbdx[k] = 0.0f;
        }
        
    /* Evaluate the polynomial and its derivative. */
    for ( i = 2 ; i >= 0 ; i-- )
        for ( k = 0 ; k < N ; k++ ) {
            dbdx[k] = dbdx[k] * x[k] + b[k];
            b[k] = b[k] * x[k] + spme_coeffs[ 4*ival[k] + i ];
            }
            
    }
    
    
/**
 * @breif Recursive definition of the cardinal B-spline
 *
 * @param k B-spline order.
 * @param x Point at which the function will be evaluated.
 */
 
float spme_M ( int k , float x ) {

#ifdef HAVE_FFTW3
    /* Lowest order? */
    if ( k == 1 ) {
    
        /* In the house? */
        if ( x >= 0.0f && x <= 1.0f )
            return 1.0f;
        else
            return 0.0f;
    
        }
        
    /* Otherwise, recurse. */
    else
        return x / (k - 1) * spme_M( k-1 , x ) + (k - x) / (k - 1) * spme_M( k-1 , x - 1.0f );
#else
    return error(smpe_err_nofftw3);
#endif

    }
    
    
/**
 * @brief Sort the cell pairs in a spme according to their direction.
 *
 * @param s The #spme data structure.
 * @param dim The SPME grid dimensions.
 * @param h The grid spacing in each dimension.
 *
 * @return #spme_err_ok or < 0 on error (see #spme_err).
 */
 
int spme_init ( struct spme *s , int *dim , float *h , float kappa ) {

#ifdef HAVE_FFTW3
    int i, j, k, l, size;
    float v, m[3], m2, kappa2 = kappa*kappa;
    fftwf_complex bc[3];
    float b[3], w[3];
    float x[ spme_order-1 ], mx[ spme_order-1 ], dmdx[ spme_order-1 ];

    /* Sanity check. */
    if ( s == NULL || dim == NULL || h == NULL )
        return error(spme_err_null);

    /* Set the dimensions. */
    s->dim[0] = dim[0];
    s->dim[1] = dim[1];
    s->dim[2] = dim[2];
    
    /* Set the spacing. */
    s->h[0] = h[0]; s->ih[0] = 1.0f / h[0]; w[0] = s->ih[0] / dim[0];
    s->h[1] = h[1]; s->ih[1] = 1.0f / h[1]; w[1] = s->ih[1] / dim[1];
    s->h[2] = h[2]; s->ih[2] = 1.0f / h[2]; w[2] = s->ih[2] / dim[2];
    
    /* Store kappa. */
    s->kappa = kappa;
    
    /* Allocate the grids. */
    size = dim[0] * dim[1] * dim[2];
    if ( posix_memalign( (void **)&s->Q , 16 , sizeof(fftwf_complex) * size ) != 0 ||
         posix_memalign( (void **)&s->T , 16 , sizeof(fftwf_complex) * size ) != 0 ||
         posix_memalign( (void **)&s->theta , 16 , sizeof(float) * size ) != 0 ||
         posix_memalign( (void **)&s->E , 16 , sizeof(fftwf_complex) * size ) != 0 )
        return error(spme_err_malloc);
        
    /* Init the grids. */
    bzero( s->Q , sizeof(float) * size );
    bzero( s->E , sizeof(float) * size );
    
    /* Pre-compute the b-splines at the integers. */
    for ( k = 0 ; k <= spme_order-2 ; k++ )
        x[k] = abs( k + 1 - spme_order/2 );
    spme_bspline( x , spme_order-1 , mx , dmdx );
        
    /* Fill theta (reciprocal space). */
    v = 1.0f / ( M_PI * dim[0]*h[0] * dim[1]*h[1] * dim[2]*h[2] );
    for ( i = 0 ; i < dim[0] ; i++ )
        for ( j = 0 ; j < dim[1] ; j++ )
            for ( k = 0 ; k < dim[2] ; k++ ) {
            
                /* Compute the Euler exponential splines. */
                bc[0] = 0.0; bc[1] = 0.0; bc[2] = 0.0;
                for ( l = 0 ; l <= spme_order-2 ; l++ ) {
                    bc[0] += mx[l] * cexpf( 2*M_PI*i*l/dim[0]*I );
                    bc[1] += mx[l] * cexpf( 2*M_PI*j*l/dim[1]*I );
                    bc[2] += mx[l] * cexpf( 2*M_PI*k*l/dim[2]*I );
                    }
                b[0] = cabsf( cexpf( 2*M_PI*I*(spme_order-1)*i/dim[0] ) / bc[0] );
                b[1] = cabsf( cexpf( 2*M_PI*I*(spme_order-1)*j/dim[1] ) / bc[1] );
                b[2] = cabsf( cexpf( 2*M_PI*I*(spme_order-1)*k/dim[2] ) / bc[2] );
                
                /* Get the corrected values for m. */
                m[0] = ( (i > dim[0]/2) ? i - dim[0] : i ) * w[0];
                m[1] = ( (j > dim[1]/2) ? j - dim[1] : j ) * w[1];
                m[2] = ( (k > dim[2]/2) ? k - dim[2] : k ) * w[2];
                m2 = m[0]*m[0] + m[1]*m[1] + m[2]*m[2];
                
                /* Fill theta. */
                s->theta[ k + dim[2]*(j + dim[1]*i) ] = b[0]*b[0] * b[1]*b[1] * b[2]*b[2] * v * expf( -M_PI*M_PI / kappa2 * m2 ) / m2;
                
                /* Catch problems on the grid. */
                if ( !isfinite( s->theta[ k + dim[2]*(j + dim[1]*i) ] ) )
                    s->theta[ k + dim[2]*(j + dim[1]*i) ] = 0.0f;
                
                }
                
    /* Init the plans. */
    if ( ( s->fwplan = fftwf_plan_dft_3d( dim[0] , dim[1] , dim[2] , s->Q , s->T , FFTW_FORWARD , FFTW_ESTIMATE ) ) == NULL )
        return error(spme_err_fftw);
    if ( ( s->bwplan = fftwf_plan_dft_3d( dim[0] , dim[1] , dim[2] , s->T , s->E , FFTW_BACKWARD , FFTW_ESTIMATE ) ) == NULL )
        return error(spme_err_fftw);
                
    /* We're on a roll. */
    return spme_err_ok;
#else
    return error(smpe_err_nofftw3);
#endif

    }

