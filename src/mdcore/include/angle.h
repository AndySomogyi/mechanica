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
#ifndef INCLUDE_ANGLE_H
#define INCLUDE_ANGLE_H
#include "platform.h"

/* Include configuration header */
#include "mdcore_config.h"



/* angle error codes */
#define angle_err_ok                     0
#define angle_err_null                  -1
#define angle_err_malloc                -2


/** ID of the last error */
CAPI_DATA(int) angle_err;


typedef enum MxAngleFlags {

    // none type angles are initial state, and can be
    // re-assigned if ref count is 1 (only owned by engine).
    ANGLE_NONE                   = 0,
    ANGLE_ACTIVE                 = 1 << 0,
    ANGLE_FOO   = 1 << 1,
} MxAngleFlags;


/** The angle structure */
typedef struct MxAngle : PyObject {

    uint32_t flags;

	/* ids of particles involved */
	int i, j, k;

	/* id of the potential. */
	struct MxPotential *potential;

} MxAngle;


/**
 * @brief Add a angle interaction to the engine.
 *
 * @param i The ID of the first #part.
 * @param j The ID of the second #part.
 * @param k The ID of the third #part.
 * @param pid Index of the #potential for this bond.
 *
 * Note, the potential (pid) has to be previously added by engine_angle_addpot.
 */
CAPI_FUNC(MxAngle*) MxAngle_NewFromIds(int i , int j , int k , int pid );

/**
 * @brief Add a angle interaction to the engine.
 *
 * @param i The ID of the first #part.
 * @param j The ID of the second #part.
 * @param k The ID of the third #part.
 * @param pot An existing potential.
 *
 * This checks if the potential is already in the engine, and if so, uses it,
 * otherwise, adds the potential to the engine.
 */
CAPI_FUNC(MxAngle*) MxAngle_NewFromIdsAndPotential(int i , int j , int k , struct MxPotential *pot);


/**
 * Internal function to initialize the angle Python api.
 */
HRESULT _MxAngle_init(PyObject *module);



/* associated functions */
int angle_eval ( struct MxAngle *a , int N , struct engine *e , double *epot_out );
int angle_evalf ( struct MxAngle *a , int N , struct engine *e , FPTYPE *f , double *epot_out );


#endif // INCLUDE_ANGLE_H
