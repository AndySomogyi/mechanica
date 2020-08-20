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

CAPI_STRUCT(NOMStyle);


/* error codes */
#define PARTICLE_ERR_OK                 0
#define PARTICLE_ERR_NULL              -1
#define PARTICLE_ERR_MALLOC            -2

typedef enum MxParticleTypeFlags {
    PARTICLE_TYPE_NONE          = 0,
    PARTICLE_TYPE_INERTIAL      = 1 << 0,
    PARTICLE_TYPE_DISSAPATIVE   = 1 << 1,
} MxParticleTypeFlags;

typedef enum MxParticleDynamics {
    PARTICLE_NEWTONIAN          = 0,
    PARTICLE_OVERDAMPED           = 1,
} MxParticleDynamics;

/* particle flags */
typedef enum MxParticleFlags {
    PARTICLE_FLAG_NONE          = 0,
    PARTICLE_FLAG_FROZEN        = 1 << 0,
    PARTICLE_FLAG_GHOST         = 1 << 1,
} MxParticleFlags;



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

        struct {
            float __dummy[3];
            uint32_t creation_time;
        };
    };

	/** Particle velocity */
    union {
        FPTYPE v[4] __attribute__ ((aligned (16)));
        Magnum::Vector3 velocity __attribute__ ((aligned (16))) = {0,0,0};
    };

	/**
	 * Particle force
	 *
	 * ONLY the coherent part of the force should go here. We use multi-step
	 * integrators, that need to separate the random and coherent forces.
	 */
    union {
        FPTYPE f[4] __attribute__ ((aligned (16)));
        Magnum::Vector3 force __attribute__ ((aligned (16))) = {0,0,0};
    };
    
    /** random force force */
    union {
        Magnum::Vector3 persistent_force __attribute__ ((aligned (16))) = {0,0,0};
    };
    

    // inverse mass
    double imass;

    float radius;
    
    double mass;

	/** individual particle charge, if needed. */
	float q;

	// runge-kutta k intermediates.
	Magnum::Vector3 p0;
	Magnum::Vector3 v0;
	Magnum::Vector3 xk[4];
	Magnum::Vector3 vk[4];
    

	/** 
	 * Particle id, virtual id 
	 * TODO: not sure what virtual id is...
	 */
	int id, vid;

	/** particle type. */
	short int typeId;

	/** Particle flags */
	unsigned short int flags;
    
    /**
     * pointer to the python 'wrapper'. Need this because the particle data
     * gets moved around between cells, and python can't hold onto that directly,
     * so keep a pointer to the python object, and update that pointer
     * when this object gets moved.
     */
    struct MxPyParticle *pyparticle;
    
    
    // style pointer, set at object construction time.
    // may be re-set by users later.
    // the base particle type has a default style. 
    NOMStyle *style;
};


/**
 * Layout of the actual Python particle object.
 *
 * The engine allocates particle memory in blocks, and particle
 * values get moved around all the time, so their addresses change.
 *
 * The partlist is always ordered  by id, i.e. partlist[id]  always
 * points to the same particle, even though that particle may move
 * from cell to cell.
 */
struct MxPyParticle : PyObject {
    int id;
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
    
    /**
     * particle type flags
     */
    uint32_t flags;
    
    
    /** Constant physical characteristics */
    double mass, imass, charge;
    
    /** default radius for particles that don't define their own radius */
    double radius;
    
    /**
     * energy and potential energy of this type, this is updated by the engine
     * each time step.
     */
    double kinetic_energy;
    
    double potential_energy;
    
    double target_energy;
    
    /**
     * minimum radius, if a fission event occurs, it will not spit a particle
     * such that it's radius gets less than this value.
     *
     * defaults to radius
     */
    double minumum_radius;

    /** Nonbonded interaction parameters. */
    double eps, rmin;
    
    /**
     * what kind of propator does this particle type use?
     */
    unsigned char dynamics;
    
    /** Name of this particle type. */
    char name[MAX_NAME], name2[MAX_NAME];
    
    /** number of current particles of this type. Incremented in engine_addpart. */
    uint32_t nr_parts = 0;
    
    /**
     * list of particle ids that are of this type
     */
    int32_t *part_ids;
    
    // max size of the ids array. 
    uint32_t size_parts;
    
    
    // style pointer, optional.
    NOMStyle *style;
    
    /**
     * add a particle (id) to this type
     */
    HRESULT addpart(int32_t id);
    
    
    /**
     * remove a particle id from this type
     */
    HRESULT del_part(int32_t id);
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
CAPI_FUNC(int) MxParticle_Check(PyObject *o);


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
 * simple fission,
 *
 * divides a particle into two, and creates a new daughter particle in the
 * universe.
 *
 * Vector of numbers indicate how to split the attached chemical cargo.
 */
CAPI_FUNC(PyObject*) MxParticle_FissionSimple(MxParticle *part,
        MxParticleType *a, MxParticleType *b,
        int nPartitionRatios, float *partitionRations);



/**
 * internal function to initalize the particle and particle types
 */
HRESULT _MxParticle_init(PyObject *m);

#endif // INCLUDE_PARTICLE_H_
