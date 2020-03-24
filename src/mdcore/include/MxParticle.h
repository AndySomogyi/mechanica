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
#ifndef INCLUDE_PARTICLE_H_
#define INCLUDE_PARTICLE_H_
#include "platform.h"
#include "fptype.h"
#include "carbon.h"

/* error codes */
#define PARTICLE_ERR_OK                 0
#define PARTICLE_ERR_NULL              -1
#define PARTICLE_ERR_MALLOC            -2


/* particle flags */
#define PARTICLE_FLAG_NONE              0
#define PARTICLE_FLAG_FROZEN            1
#define PARTICLE_FLAG_GHOST             2


/* default values */

MDCORE_BEGIN_DECLS


/** ID of the last error. */
extern int particle_err;


/**
 * The particle data structure.
 *
 * Instance vars for each particle.
 *
 * Note that the arrays for @c x, @c v and @c f are 4 entries long for
 * propper alignment.
 */
struct MxParticle : PyObject  {

	/** Particle position */
	FPTYPE x[4] __attribute__ ((aligned (16)));

	/** Particle velocity */
	FPTYPE v[4] __attribute__ ((aligned (16)));

	/** Particle force */
	FPTYPE f[4] __attribute__ ((aligned (16)));

	/** individual particle charge, if needed. */
	float q;

	/** Particle id and type */
	int id, vid;

	/** particle type. */
	short int typeId;

	/** Particle flags */
	unsigned short int flags;

	PyObject *pos;

	PyObject *vel;

	PyObject *force;


};





/**
 * Structure containing information on each particle species.
 *
 * This is only a definition for the particle *type*, not the actual
 * instance vars like pos, vel, which are stored in part.
 *
 * Extend the PyHeapTypeObject, because this is the actual type that
 * gets allocated, its a python thing.
 */
struct MxParticleType : PyHeapTypeObject {

	/** ID of this type */
	int id;

	/** Constant physical characteristics */
	double mass, imass, charge;

	/** Nonbonded interaction parameters. */
	double eps, rmin;

	/** Name of this paritcle type. */
	char name[64], name2[64];

} ;

/**
 * The type of each individual particle.
 */
CAPI_DATA(MxParticleType) MxParticle_Type;

/**
 * The the particle type type
 */
CAPI_DATA(PyTypeObject) MxParticleType_Type;

/**
 * Determines if this object is a particle type.
 * @returns TRUE if a symbol, FALSE otherwise.
 */
CAPI_FUNC(int) MxParticleCheck(PyObject *o);


/* associated functions */
int md_particle_init ( struct MxParticle *p , int vid , int type , unsigned int flags );


/**
 * internal function to initalize the particle and particle types
 */
HRESULT MxParticle_init(PyObject *m);

MDCORE_END_DECLS
#endif // INCLUDE_PARTICLE_H_
