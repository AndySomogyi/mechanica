/*******************************************************************************
 * This file is part of mdcore.
 * Coypright (c) 2012 Pedro Gonnet (pedro.gonnet@durham.ac.uk)
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
#include <pthread.h>
#include <math.h>
#include <float.h>
#include <string.h>

/* Include conditional headers. */
#include "../config.h"
#ifdef HAVE_MPI
    #include <mpi.h>
#endif
#ifdef HAVE_OPENMP
    #include <omp.h>
#endif

/* Disable vectorization for the nvcc compiler's sake. */
#undef __SSE__
#undef __SSE2__
#undef __ALTIVEC__
#undef __AVX__

/* include local headers */
#include "cycle.h"
#include "errs.h"
#include "fptype.h"
#include "lock.h"
#include "part.h"
#include "cell.h"
#include "task.h"
#include "space.h"
#include "potential.h"
#include "bond.h"
#include "rigid.h"
#include "angle.h"
#include "dihedral.h"
#include "exclusion.h"
#include "reader.h"
#include "engine.h"
#include "runner_cuda.h"


/* the error macro. */
#define error(id)				( engine_err = errs_register( id , engine_err_msg[-(id)] , __LINE__ , __FUNCTION__ , __FILE__ ) )
#define cuda_error(id)			( engine_err = errs_register( id , cudaGetErrorString(cudaGetLastError()) , __LINE__ , __FUNCTION__ , __FILE__ ) )

/* As of here there is only CUDA-related stuff. */
#ifdef HAVE_CUDA


/* The parts (non-texture access). */
extern __constant__ float4 *cuda_parts;

/* Forward declaration of runner kernel. */
__global__ void runner_run_verlet_cuda_32 ( float *forces , int *counts , int *ind , int verlet_rebuild );
__global__ void runner_run_cuda_32 ( float *forces , int *counts , int *ind );
__global__ void runner_run_verlet_cuda_64 ( float *forces , int *counts , int *ind , int verlet_rebuild );
__global__ void runner_run_cuda_64 ( float *forces , int *counts , int *ind );
__global__ void runner_run_verlet_cuda_96 ( float *forces , int *counts , int *ind , int verlet_rebuild );
__global__ void runner_run_cuda_96 ( float *forces , int *counts , int *ind );
__global__ void runner_run_verlet_cuda_128 ( float *forces , int *counts , int *ind , int verlet_rebuild );
__global__ void runner_run_cuda_128 ( float *forces , int *counts , int *ind );
__global__ void runner_run_verlet_cuda_160 ( float *forces , int *counts , int *ind , int verlet_rebuild );
__global__ void runner_run_cuda_160 ( float *forces , int *counts , int *ind );
__global__ void runner_run_verlet_cuda_192 ( float *forces , int *counts , int *ind , int verlet_rebuild );
__global__ void runner_run_cuda_192 ( float *forces , int *counts , int *ind );
__global__ void runner_run_verlet_cuda_224 ( float *forces , int *counts , int *ind , int verlet_rebuild );
__global__ void runner_run_cuda_224 ( float *forces , int *counts , int *ind );
__global__ void runner_run_verlet_cuda_256 ( float *forces , int *counts , int *ind , int verlet_rebuild );
__global__ void runner_run_cuda_256 ( float *forces , int *counts , int *ind );
__global__ void runner_run_verlet_cuda_288 ( float *forces , int *counts , int *ind , int verlet_rebuild );
__global__ void runner_run_cuda_288 ( float *forces , int *counts , int *ind );
__global__ void runner_run_verlet_cuda_320 ( float *forces , int *counts , int *ind , int verlet_rebuild );
__global__ void runner_run_cuda_320 ( float *forces , int *counts , int *ind );
__global__ void runner_run_verlet_cuda_352 ( float *forces , int *counts , int *ind , int verlet_rebuild );
__global__ void runner_run_cuda_352 ( float *forces , int *counts , int *ind );
__global__ void runner_run_verlet_cuda_384 ( float *forces , int *counts , int *ind , int verlet_rebuild );
__global__ void runner_run_cuda_384 ( float *forces , int *counts , int *ind );
__global__ void runner_run_verlet_cuda_416 ( float *forces , int *counts , int *ind , int verlet_rebuild );
__global__ void runner_run_cuda_416 ( float *forces , int *counts , int *ind );
__global__ void runner_run_verlet_cuda_448 ( float *forces , int *counts , int *ind , int verlet_rebuild );
__global__ void runner_run_cuda_448 ( float *forces , int *counts , int *ind );
__global__ void runner_run_verlet_cuda_480 ( float *forces , int *counts , int *ind , int verlet_rebuild );
__global__ void runner_run_cuda_480 ( float *forces , int *counts , int *ind );
// __global__ void runner_run_verlet_cuda_512 ( float *forces , int *counts , int *ind , int verlet_rebuild );
// __global__ void runner_run_cuda_512 ( float *forces , int *counts , int *ind );
int runner_bind ( cudaArray *cuArray_coeffs , cudaArray *cuArray_pind , cudaArray *cuArray_diags );
int runner_parts_bind ( cudaArray *cuArray_parts );
int runner_parts_unbind ( );


/**
 * @brief Set the ID of the CUDA device to use
 *
 * @param id The CUDA device ID.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
 
extern "C" int engine_cuda_setdevice ( struct engine *e , int id ) {

    /* Store the single device ID in the engine. */
    e->nr_devices = 1;
    e->devices[0] = id;

    /* Make sure the device works, init a stream on it. */
    if ( cudaSetDevice( id ) != cudaSuccess ||
         cudaStreamCreate( (cudaStream_t *)&e->streams[0] ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    else
        return engine_err_ok;
        
    }
    

/**
 * @brief Set the number of CUDA devices to use, as well as their IDs.
 *
 * @param id The CUDA device ID.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
 
extern "C" int engine_cuda_setdevices ( struct engine *e , int nr_devices , int *ids ) {

    int k;
    
    /* Sanity check. */
    if ( nr_devices > engine_maxgpu )
        return error(engine_err_range);

    /* Store the single device ID in the engine. */
    e->nr_devices = nr_devices;
    for ( k = 0 ; k < nr_devices ; k++ ) {
    
        /* Store the device ID. */
        e->devices[k] = ids[k];

        /* Make sure the device works, init a stream on it. */
        if ( cudaSetDevice( ids[k] ) != cudaSuccess ||
             cudaStreamCreate( (cudaStream_t *)&e->streams[k] ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
            
        }
        
    /* That's it. */
    return engine_err_ok;
        
    }
    

/* End CUDA-related stuff. */
#endif
