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
#ifndef INCLUDE_ENGINE_H_
#define INCLUDE_ENGINE_H_

#include "carbon.h"
#include "platform.h"
#include "pthread.h"
#include "space.h"
#include "cycle.h"


/* engine error codes */
#define engine_err_ok                    0
#define engine_err_null                  -1
#define engine_err_malloc                -2
#define engine_err_space                 -3
#define engine_err_pthread               -4
#define engine_err_runner                -5
#define engine_err_range                 -6
#define engine_err_cell                  -7
#define engine_err_domain                -8
#define engine_err_nompi                 -9
#define engine_err_mpi                   -10
#define engine_err_bond                  -11
#define engine_err_angle                 -12
#define engine_err_reader                -13
#define engine_err_psf                   -14
#define engine_err_pdb                   -15
#define engine_err_cpf                   -16
#define engine_err_potential             -17
#define engine_err_exclusion             -18
#define engine_err_sets                  -19
#define engine_err_dihedral              -20
#define engine_err_cuda                  -21
#define engine_err_nocuda                -22
#define engine_err_cudasp                -23
#define engine_err_maxparts              -24
#define engine_err_queue                 -25
#define engine_err_rigid                 -26
#define engine_err_cutoff		 		 -27
#define engine_err_nometis				 -28
#define engine_err_toofast               -29


/* some constants */
enum EngineFlags {
    engine_flag_none                 = 0,
    engine_flag_static               = 1 << 0,
    engine_flag_localparts           = 1 << 1,
    engine_flag_cuda                 = 1 << 2,
    engine_flag_explepot             = 1 << 3,
    engine_flag_verlet               = 1 << 4,
    engine_flag_verlet_pairwise      = 1 << 5,
    engine_flag_affinity             = 1 << 6,
    engine_flag_prefetch             = 1 << 7,
    engine_flag_verlet_pseudo        = 1 << 8,
    engine_flag_unsorted             = 1 << 9,
    engine_flag_shake                = 1 << 10,
    engine_flag_mpi                  = 1 << 11,
    engine_flag_parbonded            = 1 << 12,
    engine_flag_async                = 1 << 13,
    engine_flag_sets                 = 1 << 14,
    engine_flag_nullpart             = 1 << 15,
    engine_flag_initialized          = 1 << 16,
    engine_flag_velocity_clamp       = 1 << 17,
};

enum EngineIntegrator {
    FORWARD_EULER,
    RUNGE_KUTTA_4
};

#define engine_bonds_chunk               100
#define engine_angles_chunk              100
#define engine_rigids_chunk              50
#define engine_dihedrals_chunk           100
#define engine_exclusions_chunk          100
#define engine_readbuff                  16384
#define engine_maxgpu                    10
#define engine_pshake_steps              20
#define engine_maxKcutoff                2

#define engine_split_MPI		1
#define engine_split_GPU		2

#define engine_bonded_maxnrthreads       16
#define engine_bonded_nrthreads          ((omp_get_num_threads()<engine_bonded_maxnrthreads)?omp_get_num_threads():engine_bonded_maxnrthreads)

/** Timmer IDs. */
enum {
	engine_timer_step = 0,
	engine_timer_prepare,
	engine_timer_verlet,
	engine_timer_exchange1,
	engine_timer_nonbond,
	engine_timer_bonded,
	engine_timer_bonded_sort,
	engine_timer_bonds,
	engine_timer_angles,
	engine_timer_dihedrals,
	engine_timer_exclusions,
	engine_timer_advance,
	engine_timer_rigid,
	engine_timer_exchange2,
	engine_timer_shuffle,
	engine_timer_cuda_load,
	engine_timer_cuda_unload,
	engine_timer_cuda_dopairs,
	engine_timer_last
};


/** Timmer IDs. */
enum {
    ENGINE_TIMER_STEP           = 1 << 0,
    ENGINE_TIMER_PREPARE        = 1 << 1,
    ENGINE_TIMER_VERLET         = 1 << 2,
    ENGINE_TIMER_EXCHANGE1      = 1 << 3,
    ENGINE_TIMER_NONBOND        = 1 << 4,
    ENGINE_TIMER_BONDED         = 1 << 5,
    ENGINE_TIMER_BONDED_SORT    = 1 << 6,
    ENGINE_TIMER_BONDS          = 1 << 7,
    ENGINE_TIMER_ANGLES         = 1 << 8,
    ENGINE_TIMER_DIHEDRALS      = 1 << 9,
    ENGINE_TIMER_EXCLUSIONS     = 1 << 10,
    ENGINE_TIMER_ADVANCE        = 1 << 11,
    ENGINE_TIMER_RIGID          = 1 << 12,
    ENGINE_TIMER_EXCHANGE2      = 1 << 13,
    ENGINE_TIMER_SHUFFLE        = 1 << 14,
    ENGINE_TIMER_CUDA_LOAD      = 1 << 15,
    ENGINE_TIMER_CUDA_UNLOAD    = 1 << 16,
    ENGINE_TIMER_CUDA_DOPAIRS   = 1 << 17,
    ENGINE_TIMER_LAST           = 1 << 18
};

enum {
    // forces that set the persistent_force should
    // update values now. Otherwise, the integrator is
    // probably in a multi-step and should use the saved
    // value
    INTEGRATOR_UPDATE_PERSISTENTFORCE    = 1 << 0
};


/** ID of the last error. */
CAPI_DATA(int) engine_err;

/** List of error messages. */
CAPI_DATA(const char *) engine_err_msg[];


/** 
 * The #engine structure. 
 */
typedef struct engine {

	/** Some flags controlling how this engine works. */
	unsigned int flags;

	/**
	 * Internal flags related to multi-step integrators,
	 */
	unsigned int integrator_flags;

#ifdef WITH_CUDA
	/** Some flags controlling which cuda scheduling we use. */
	unsigned int flags_cuda;
#endif

	/** The space on which to work */
	struct space s;

	/** Time variables */
	long time;
	double dt;

	double temperature;

	// Boltzmann constant
	double K;

	/** TODO, clean up this design for types and static engine. */
	/** What is the maximum nr of types? */
	static int max_type;
	static int nr_types;

	/** The particle types. */
    static struct MxParticleType *types;

	/** The interaction matrix */
	struct MxPotential **p, **p_dihedral;

	/** The explicit electrostatic potential. */
	struct MxPotential *ep;

	/**
	 * vector of single body potentials for types, indexed
	 * by type id.
	 */
	struct MxForce **p_singlebody;

	/** Mutexes, conditions and counters for the barrier */
	pthread_mutex_t barrier_mutex;
	pthread_cond_t barrier_cond;
	pthread_cond_t done_cond;
	int barrier_count;

	/** Nr of runners */
	int nr_runners;

	/** The runners */
	struct runner *runners;

	/** The queues for the runners. */
	struct queue *queues;
	int nr_queues;

	/** The ID of the computational node we are on. */
	int nodeID;
	int nr_nodes;

	/** Lists of cells to exchange with other nodes. */
	struct engine_comm *send, *recv;

	/** List of bonds. */
	struct MxBond *bonds;

	/** Nr. of bonds. */
	int nr_bonds, bonds_size;

	/** List of exclusions. */
	struct exclusion *exclusions;

	/** Nr. of exclusions. */
	int nr_exclusions, exclusions_size;

	/** List of rigid bodies. */
	struct rigid *rigids;

	/** List linking parts to rigids. */
	int *part2rigid;

	/** Nr. of rigids. */
	int nr_rigids, rigids_size, nr_constr, rigids_local, rigids_semilocal;

	/** Rigid solver tolerance. */
	double tol_rigid;

	/** List of angles. */
	struct MxAngle *angles;

	/** Nr. of angles. */
	int nr_angles, angles_size;

	/** List of dihedrals. */
	struct dihedral *dihedrals;

	/** Nr. of dihedrals. */
	int nr_dihedrals, dihedrals_size, nr_dihedralpots, dihedralpots_size;

	/** The Comm object for mpi. */
#ifdef WITH_MPI
	MPI_Comm comm;
	pthread_mutex_t xchg_mutex;
	pthread_cond_t xchg_cond;
	short int xchg_started, xchg_running;
	pthread_t thread_exchg;
	pthread_mutex_t xchg2_mutex;
	pthread_cond_t xchg2_cond;
	short int xchg2_started, xchg2_running;
	pthread_t thread_exchg2;
#endif

	/** Pointers to device data for CUDA. */
#ifdef HAVE_CUDA
	void *sortlists_cuda[ engine_maxgpu ];
	int nrpots_cuda, *pind_cuda[ engine_maxgpu ], *offsets_cuda[ engine_maxgpu ];
	int nr_devices, devices[ engine_maxgpu ];
	float *forces_cuda[ engine_maxgpu ];
	void *cuArray_parts[ engine_maxgpu ], *parts_cuda[ engine_maxgpu ];
	void *parts_cuda_local;
	int *cells_cuda_local[ engine_maxgpu];
	int cells_cuda_nr[ engine_maxgpu ];
	int *counts_cuda[ engine_maxgpu ], *counts_cuda_local[ engine_maxgpu ];
	int *ind_cuda[ engine_maxgpu ], *ind_cuda_local[ engine_maxgpu ];
	struct task_cuda *tasks_cuda[ engine_maxgpu ];
	int *taboo_cuda[ engine_maxgpu ];
	int nrtasks_cuda[ engine_maxgpu ];
	void *streams[ engine_maxgpu ];
#endif

	/** Timers. */
	ticks timers[engine_timer_last];

	/** Bonded sets. */
	struct engine_set *sets;
	int nr_sets;

	struct CMulticastTimeEvent *on_time;

	EngineIntegrator integrator;
} engine;


/**
 * Structure storing grouped sets of bonded interactions.
 */
typedef struct engine_set {

	/* Counts of the different interaction types. */
	int nr_bonds, nr_angles, nr_dihedrals, nr_exclusions, weight;

	/* Lists of ID of the relevant bonded types. */
	struct MxBond *bonds;
	struct MxAngle *angles;
	struct dihedral *dihedrals;
	struct exclusion *exclusions;

	/* Nr of sets with which this set conflicts. */
	int nr_confl;

	/* IDs of the sets with which this set conflicts. */
	int *confl;

} engine_set;


/**
 * Structure storing which cells to send/receive to/from another node.
 */
typedef struct engine_comm {

	/* Size and count of the cellids. */
	int count, size;

	int *cellid;

} engine_comm;


/* associated functions */
CAPI_FUNC(int) engine_addpot ( struct engine *e , struct MxPotential *p , int i , int j );
CAPI_FUNC(int) engine_addforce1 ( struct engine *e , struct MxForce *p , int i );
CAPI_FUNC(int) engine_advance ( struct engine *e );


/**
 * allocates a new angle, returns a pointer to it.
 */
CAPI_FUNC(int) engine_angle_alloc (struct engine *e , PyTypeObject *type, struct MxAngle **result );

/**
 * @brief Add a angle potential.
 *
 * @param e The #engine.
 * @param p The #potential to add to the #engine.
 *
 * @return The ID of the added angle potential or < 0 on error (see #engine_err).
 */
//CAPI_FUNC(int) engine_angle_addpot ( struct engine *e , struct MxPotential *p );

/**
 * @brief Add a angle interaction to the engine.
 *
 * @param e The #engine.
 * @param i The ID of the first #part.
 * @param j The ID of the second #part.
 * @param k The ID of the third #part.
 * @param pid Index of the #potential for this bond.
 *
 * Note, the potential (pid) has to be previously added by engine_angle_addpot.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
CAPI_FUNC(int) engine_angle_add ( struct engine *e , int i , int j , int k , int pid );


CAPI_FUNC(int) engine_angle_eval ( struct engine *e );
CAPI_FUNC(int) engine_barrier ( struct engine *e );


CAPI_FUNC(int) engine_bond_eval ( struct engine *e );
CAPI_FUNC(int) engine_bonded_eval ( struct engine *e );
CAPI_FUNC(int) engine_bonded_eval_sets ( struct engine *e );
CAPI_FUNC(int) engine_bonded_sets ( struct engine *e , int max_sets );
CAPI_FUNC(int) engine_dihedral_add ( struct engine *e , int i , int j , int k , int l , int pid );
CAPI_FUNC(int) engine_dihedral_addpot ( struct engine *e , struct MxPotential *p );
CAPI_FUNC(int) engine_dihedral_eval ( struct engine *e );
CAPI_FUNC(int) engine_dump_PSF ( struct engine *e , FILE *psf , FILE *pdb , char *excl[] , int nr_excl );
CAPI_FUNC(int) engine_exclusion_add ( struct engine *e , int i , int j );
CAPI_FUNC(int) engine_exclusion_eval ( struct engine *e );
CAPI_FUNC(int) engine_exclusion_shrink ( struct engine *e );
CAPI_FUNC(int) engine_finalize ( struct engine *e );
CAPI_FUNC(int) engine_flush_ghosts ( struct engine *e );
CAPI_FUNC(int) engine_flush ( struct engine *e );
CAPI_FUNC(int) engine_gettype ( struct engine *e , char *name );
CAPI_FUNC(int) engine_gettype2 ( struct engine *e , char *name2 );

/**
 * allocates a new bond, returns a pointer to it.
 */
int engine_bond_alloc (struct engine *e , struct _typeobject *type, struct MxBond **result );

/**
 * External C apps should call this to get a particle type ptr.
 */
CAPI_FUNC(struct MxParticleType*) engine_type(int id);

/**
 * @brief Add a #part to a #space at the given coordinates. The given
 * particle p is only used for the attributes, it itself is not added,
 * rather a new memory block is allocated, and the contents of p
 * get copied in there.
 *
 * @param s The space to which @c p should be added.
 * @param p The #part to be added.
 * @param x A pointer to an array of three doubles containing the particle
 *      position.
 * @param result pointer to the newly allocated particle.
 *
 * @returns #space_err_ok or < 0 on error (see #space_err).
 *
 * Inserts a #part @c p into the #space @c s at the position @c x.
 * Note that since particle positions in #part are relative to the cell, that
 * data in @c p is overwritten and @c x is used.
 *
 * This is the single, central function that actually allocates particle space,
 * and inserts a new particle into the engine.
 *
 * Increases the ref count on the particle type.
 */
CAPI_FUNC(int) engine_addpart ( struct engine *e ,  struct MxParticle *p ,
        double *x, struct MxParticle **result );

/**
 * Adds a force for a given type id
 *
 * The engine 'borrows' (increases the ref count) to the force.
 *
 * @param e: engine ptr
 * @param f: ptr to force
 * @param id: id of particle type.
 */
CAPI_FUNC(int) engine_singlebody_set (struct engine *e , struct MxForce *f, int type_id);


/**
 * @brief Add a type definition.
 *
 * @param e The #engine.
 * @param mass The particle type mass.
 * @param charge The particle type charge.
 * @param name Particle name, can be @c NULL.
 * @param name2 Particle second name, can be @c NULL.
 *
 * @return The type ID or < 0 on error (see #engine_err).
 *
 * The particle type ID must be an integer greater or equal to 0
 * and less than the value @c max_type specified in #engine_init.
 */
CAPI_FUNC(int) engine_addtype ( struct engine *e , double mass , double charge ,
        const char *name , const char *name2 );


/**
 * @brief Initialize an #engine with the given data.
 *
 * The number of spatial cells in each cartesion dimension is floor( dim[i] / L[i] ), or
 * the physical size of the space in that dimension divided by the minimum size size of
 * each cell.
 *
 * @param e The #engine to initialize.
 * @param origin An array of three doubles containing the cartesian origin
 *      of the space.
 * @param dim An array of three doubles containing the size of the space.
 *
 * @param L The minimum spatial cell edge length in each dimension.
 *
 * @param cutoff The maximum interaction cutoff to use.
 * @param period A bitmask describing the periodicity of the domain
 *      (see #space_periodic_full).
 * @param max_type The maximum number of particle types that will be used
 *      by this engine.
 * @param flags Bit-mask containing the flags for this engine.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
CAPI_FUNC(int) engine_init ( struct engine *e , const double *origin , const double *dim , double *L ,
		double cutoff , unsigned int period , int max_type , unsigned int flags );


CAPI_FUNC(int) engine_load_ghosts ( struct engine *e , double *x , double *v , int *type , int *pid ,
		int *vid , double *q , unsigned int *flags , int N );
CAPI_FUNC(int) engine_load ( struct engine *e , double *x , double *v , int *type , int *pid , int *vid ,
		double *charge , unsigned int *flags , int N );
CAPI_FUNC(int) engine_nonbond_eval ( struct engine *e );
CAPI_FUNC(int) engine_read_cpf ( struct engine *e , int cpf , double kappa , double tol , int rigidH );
CAPI_FUNC(int) engine_read_psf ( struct engine *e , int psf , int pdb );
CAPI_FUNC(int) engine_read_xplor ( struct engine *e , int xplor , double kappa , double tol , int rigidH );
CAPI_FUNC(int) engine_rigid_add ( struct engine *e , int pid , int pjd , double d );
CAPI_FUNC(int) engine_rigid_eval ( struct engine *e );
CAPI_FUNC(int) engine_rigid_sort ( struct engine *e );
CAPI_FUNC(int) engine_rigid_unsort ( struct engine *e );
CAPI_FUNC(int) engine_setexplepot ( struct engine *e , struct MxPotential *ep );
CAPI_FUNC(int) engine_shuffle ( struct engine *e );
CAPI_FUNC(int) engine_split_bisect ( struct engine *e , int N );
CAPI_FUNC(int) engine_split ( struct engine *e );

CAPI_FUNC(int) engine_start ( struct engine *e , int nr_runners , int nr_queues );
CAPI_FUNC(int) engine_step ( struct engine *e );
CAPI_FUNC(int) engine_timers_reset ( struct engine *e );
CAPI_FUNC(int) engine_unload_marked ( struct engine *e , double *x , double *v , int *type , int *pid ,
		int *vid , double *q , unsigned int *flags , double *epot , int N );
CAPI_FUNC(int) engine_unload_strays ( struct engine *e , double *x , double *v , int *type , int *pid ,
		int *vid , double *q , unsigned int *flags , double *epot , int N );
CAPI_FUNC(int) engine_unload ( struct engine *e , double *x , double *v , int *type , int *pid , int *vid ,
		double *charge , unsigned int *flags , double *epot , int N );
CAPI_FUNC(int) engine_verlet_update ( struct engine *e );

/**
 * gets the next available particle id to use when creating a new particle.
 */
CAPI_FUNC(int) engine_next_partid(struct engine *e);


/**
 * Deletes a particle from the engine based on particle id.
 *
 * Afterwards, the particle id will point to a null entry in the partlist.
 *
 * Note, the next newly created particle will re-use this ID (assuming
 * the engine_next_partid is used to determine the next id.)
 */
CAPI_FUNC(HRESULT) engine_del_particle(struct engine *e, int pid);


CAPI_FUNC(void) engine_dump();

#define ENGINE_DUMP(msg) {std::cout<<msg<<std::endl; engine_dump();}

CAPI_FUNC(double) engine_kinetic_energy(struct engine *e);

CAPI_FUNC(double) engine_temperature(struct engine *e);

#ifdef WITH_MPI
CAPI_FUNC(int) engine_init_mpi ( struct engine *e , const double *origin , const double *dim , double *L ,
		double cutoff , unsigned int period , int max_type , unsigned int flags , MPI_Comm comm ,
		int rank );
CAPI_FUNC(int) engine_exchange ( struct engine *e );
CAPI_FUNC(int) engine_exchange_async ( struct engine *e );
CAPI_FUNC(int) engine_exchange_async_run ( struct engine *e );
CAPI_FUNC(int) engine_exchange_incomming ( struct engine *e );
CAPI_FUNC(int) engine_exchange_rigid ( struct engine *e );
CAPI_FUNC(int) engine_exchange_rigid_async ( struct engine *e );
CAPI_FUNC(int) engine_exchange_rigid_async_run ( struct engine *e );
CAPI_FUNC(int) engine_exchange_rigid_wait ( struct engine *e );
CAPI_FUNC(int) engine_exchange_wait ( struct engine *e );
#endif

#if defined(HAVE_CUDA) && defined(WITH_CUDA)
CAPI_FUNC(int) engine_nonbond_cuda ( struct engine *e );
CAPI_FUNC(int) engine_cuda_load ( struct engine *e );
CAPI_FUNC(int) engine_cuda_load_parts ( struct engine *e );
CAPI_FUNC(int) engine_cuda_unload_parts ( struct engine *e );
CAPI_FUNC(int) engine_cuda_setdevice ( struct engine *e , int id );
CAPI_FUNC(int) engine_cuda_setdevices ( struct engine *e , int nr_devices , int *ids );
CAPI_FUNC(int) engine_split_METIS ( struct engine *e, int N, int flags);
#endif

#ifdef WITH_METIS
CAPI_FUNC(int) engine_split_METIS ( struct engine *e, int N, int flags);
#endif

/**
 * Single static instance of the md engine per process.
 *
 * Even for MPI enabled, as each MPI process will initialize the engine with different comm and rank.
 */
CAPI_DATA(engine) _Engine;

CAPI_FUNC(struct engine*) engine_get();

#endif // INCLUDE_ENGINE_H_

