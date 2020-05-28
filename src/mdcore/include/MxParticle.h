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
#include <Magnum/Magnum.h>
#include <Magnum/Math/Vector4.h>


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
CAPI_DATA(int) particle_err;


/**
 * The particle data structure.
 *
 * Instance vars for each particle.
 *
 * Note that the arrays for @c x, @c v and @c f are 4 entries long for
 * proper alignment.
 *
 * All particles are stored in a series of contiguous blocks of memory that are owned
 * by the space cells. Each space cell has a array of particle structs.
 */
struct MxParticle  {

	/** Particle position */
    union {
        FPTYPE x[4] __attribute__ ((aligned (16)));
        Magnum::Vector3 position __attribute__ ((aligned (16))) = {0,0,0};
    };

	/** Particle velocity */
    union {
        FPTYPE v[4] __attribute__ ((aligned (16)));
        Magnum::Vector3 velocity __attribute__ ((aligned (16))) = {0,0,0};
    };

	/** Particle force */
    union {
        FPTYPE f[4] __attribute__ ((aligned (16)));
        Magnum::Vector3 force __attribute__ ((aligned (16))) = {0,0,0};
    };


	/** individual particle charge, if needed. */
	float q;
    
    float volume;

	/** Particle id and type */
	int id, vid;

	/** particle type. */
	short int typeId;

	/** Particle flags */
	unsigned short int flags;
};

struct MxPyParticle : PyObject {
    MxParticle *part;
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
    
    static const int MAX_NAME = 64;
    
    /** ID of this type */
    int id;
    
    /** Constant physical characteristics */
    double mass, imass, charge;
    
    /**
     * energy and potential energy of this type, this is updated by the engine
     * each time step.
     */
    double kinetic_energy;
    
    double potential_energy;
    
    double target_energy;

    /** Nonbonded interaction parameters. */
    double eps, rmin;
    
    /** Name of this particle type. */
    char name[MAX_NAME], name2[MAX_NAME];
    
    /** number of current particles of this type. Incremented in engine_addpart. */
    unsigned count = 0;
};

typedef MxParticleType MxParticleData;

/**
 * The type of each individual particle.
 */
//CAPI_DATA(MxParticleType) MxParticle_Type;
CAPI_FUNC(MxParticleType*) MxParticle_GetType();

/**
 * initialize the base particle type in the
 *
 * sets the engine.types[0] particle.
 *
 * The engine.types array is assumed to be allocated, but not initialized.
 */
HRESULT engine_particle_base_init(PyObject *m);

/**
 * The the particle type type
 */
CAPI_DATA(PyTypeObject) MxParticleType_Type;

/**
 * Determines if this object is a particle type.
 * @returns TRUE if a symbol, FALSE otherwise.
 */
CAPI_FUNC(int) MxParticleCheck(PyObject *o);


/**
 * Creates a new MxPyParticle wrapper, and attach it to an existing
 * particle
 */
MxPyParticle* MxPyParticle_New(MxParticle *data);

/**
 *
 *
 * Call to tp_new
 * PyObject *particle_type_new(PyTypeObject *, PyObject *, PyObject *)
 * type: <class 'ParticleType'>,
 * args: (
 *     'A',
 *     (<class 'Particle'>,),
 *     {'__module__': '__main__', '__qualname__': 'A'}
 * ),
 * kwargs: <NULL>)
 *
 * Args to tp_new should be a 3-tuple, with
 * 0: string of name of object
 * 1: base classes
 * 2: dictionary
 */


/**
 * Creates a new MxParticleType for the given particle data pointer.
 *
 * This creates a matching python type for an existing particle data,
 * and is usually called when new types are created from C.
 */
MxParticleType *MxParticleType_ForEngine(struct engine *e, double mass , double charge,
                                         const char *name , const char *name2);

/**
 * Creates and initialized a new particle type, adds it to the
 * global engine
 *
 * creates both a new type, and a new data entry in the engine.
 */
MxParticleType *MxParticleType_New(const char *_name, PyObject *dict);




/**
 * internal function to initalize the particle and particle types
 */
HRESULT _MxParticle_init(PyObject *m);

MDCORE_END_DECLS
#endif // INCLUDE_PARTICLE_H_
