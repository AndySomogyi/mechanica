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
CAPI_DATA(int) bond_err;


typedef enum MxBondFlags {

    // none type bonds are initial state, and can be
    // re-assigned if ref count is 1 (only owned by engine).
    BOND_NONE                   = 0,
    BOND_ACTIVE                 = 1 << 0,
    BOND_FOO   = 1 << 1,
} MxBondFlags;


/** The bond structure */
typedef struct MxBond {

    uint32_t flags;

	/* ids of particles involved */
	int32_t i, j;
    
    uint32_t id;

    uint64_t creation_time;

	/**
	 * half life decay time for this bond.
	 */
	double half_life;

	/* potential energy required to break this bond */
	double bond_energy;

	struct MxPotential *potential;

} MxBond;

struct MxBondHandle : PyObject {
    int32_t id;
};

CAPI_FUNC(MxBondHandle*) MxBondHandle_FromId(int id);



/**
 * The type of each individual bond.
 * actually bond handle type.
 */
CAPI_DATA(PyTypeObject) MxBondHandle_Type;

HRESULT _MxBond_init(PyObject *m);

CAPI_FUNC(MxBondHandle*) MxBondHandle_New(uint32_t flags,
        int32_t i, int32_t j,
        double half_life,
        double bond_energy,
        struct MxPotential* potential);

CAPI_FUNC(PyObject*) MxBond_PairwiseNew(
        struct MxPotential* potential,
        struct MxParticleList *parts,
        float cutoff,
        PyObject *args,
        PyObject *kwds
    );

/* associated functions */
CAPI_FUNC(int) bond_eval ( struct MxBond *b , int N , struct engine *e , double *epot_out );
CAPI_FUNC(int) bond_evalf ( struct MxBond *b , int N , struct engine *e , FPTYPE *f , double *epot_out );


#endif // INCLUDE_BOND_H_
