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
#include "platform.h"
#include "pthread.h"

MDCORE_BEGIN_DECLS

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


/* some constants */
#define engine_flag_none                 0
#define engine_flag_static               1
#define engine_flag_localparts           2
#define engine_flag_cuda                 4
#define engine_flag_explepot             8
#define engine_flag_verlet               16
#define engine_flag_verlet_pairwise      32
#define engine_flag_affinity             64
#define engine_flag_prefetch             128
#define engine_flag_verlet_pseudo        256
#define engine_flag_unsorted             512
#define engine_flag_shake                1024
#define engine_flag_mpi                  2048
#define engine_flag_parbonded            4096
#define engine_flag_async                8192
#define engine_flag_sets                 16384
#define engine_flag_nullpart             32768

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


/** ID of the last error. */
extern int engine_err;

/** List of error messages. */
extern char *engine_err_msg[];


/** 
 * The #engine structure. 
 */
typedef struct engine {

	/** Some flags controlling how this engine works. */
	unsigned int flags;

#ifdef WITH_CUDA
	/** Some flags controlling which cuda scheduling we use. */
	unsigned int flags_cuda;
#endif

	/** The space on which to work */
	struct space s;

	/** Time variables */
	int time;
	double dt;

	/** What is the maximum nr of types? */
	int max_type;
	int nr_types;

	/** The particle types. */
	struct particle_type *types;

	/** The interaction matrix */
	struct potential **p, **p_bond, **p_angle, **p_dihedral;

	/** The explicit electrostatic potential. */
	struct potential *ep;

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
	struct bond *bonds;

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
	struct angle *angles;

	/** Nr. of angles. */
	int nr_angles, angles_size, nr_anglepots, anglepots_size;

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
} engine;


/**
 * Structure storing grouped sets of bonded interactions.
 */
typedef struct engine_set {

	/* Counts of the different interaction types. */
	int nr_bonds, nr_angles, nr_dihedrals, nr_exclusions, weight;

	/* Lists of ID of the relevant bonded types. */
	struct bond *bonds;
	struct angle *angles;
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
int engine_addpot ( struct engine *e , struct potential *p , int i , int j );
int engine_addtype ( struct engine *e , double mass , double charge , char *name , char *name2 );
int engine_advance ( struct engine *e );
int engine_angle_addpot ( struct engine *e , struct potential *p );
int engine_angle_add ( struct engine *e , int i , int j , int k , int pid );
int engine_angle_eval ( struct engine *e );
int engine_barrier ( struct engine *e );
int engine_bond_addpot ( struct engine *e , struct potential *p , int i , int j );
int engine_bond_add ( struct engine *e , int i , int j );
int engine_bond_eval ( struct engine *e );
int engine_bonded_eval ( struct engine *e );
int engine_bonded_eval_sets ( struct engine *e );
int engine_bonded_sets ( struct engine *e , int max_sets );
int engine_dihedral_add ( struct engine *e , int i , int j , int k , int l , int pid );
int engine_dihedral_addpot ( struct engine *e , struct potential *p );
int engine_dihedral_eval ( struct engine *e );
int engine_dump_PSF ( struct engine *e , FILE *psf , FILE *pdb , char *excl[] , int nr_excl );
int engine_exclusion_add ( struct engine *e , int i , int j );
int engine_exclusion_eval ( struct engine *e );
int engine_exclusion_shrink ( struct engine *e );
int engine_finalize ( struct engine *e );
int engine_flush_ghosts ( struct engine *e );
int engine_flush ( struct engine *e );
int engine_gettype ( struct engine *e , char *name );
int engine_gettype2 ( struct engine *e , char *name2 );
int engine_init ( struct engine *e , const double *origin , const double *dim , double *L ,
		double cutoff , unsigned int period , int max_type , unsigned int flags );
int engine_load_ghosts ( struct engine *e , double *x , double *v , int *type , int *pid ,
		int *vid , double *q , unsigned int *flags , int N );
int engine_load ( struct engine *e , double *x , double *v , int *type , int *pid , int *vid ,
		double *charge , unsigned int *flags , int N );
int engine_nonbond_eval ( struct engine *e );
int engine_read_cpf ( struct engine *e , int cpf , double kappa , double tol , int rigidH );
int engine_read_psf ( struct engine *e , int psf , int pdb );
int engine_read_xplor ( struct engine *e , int xplor , double kappa , double tol , int rigidH );
int engine_rigid_add ( struct engine *e , int pid , int pjd , double d );
int engine_rigid_eval ( struct engine *e );
int engine_rigid_sort ( struct engine *e );
int engine_rigid_unsort ( struct engine *e );
int engine_setexplepot ( struct engine *e , struct potential *ep );
int engine_shuffle ( struct engine *e );
int engine_split_bisect ( struct engine *e , int N );
int engine_split ( struct engine *e );

int engine_start ( struct engine *e , int nr_runners , int nr_queues );
int engine_step ( struct engine *e );
int engine_timers_reset ( struct engine *e );
int engine_unload_marked ( struct engine *e , double *x , double *v , int *type , int *pid ,
		int *vid , double *q , unsigned int *flags , double *epot , int N );
int engine_unload_strays ( struct engine *e , double *x , double *v , int *type , int *pid ,
		int *vid , double *q , unsigned int *flags , double *epot , int N );
int engine_unload ( struct engine *e , double *x , double *v , int *type , int *pid , int *vid ,
		double *charge , unsigned int *flags , double *epot , int N );
int engine_verlet_update ( struct engine *e );

#ifdef WITH_MPI
int engine_init_mpi ( struct engine *e , const double *origin , const double *dim , double *L ,
		double cutoff , unsigned int period , int max_type , unsigned int flags , MPI_Comm comm ,
		int rank );
int engine_exchange ( struct engine *e );
int engine_exchange_async ( struct engine *e );
int engine_exchange_async_run ( struct engine *e );
int engine_exchange_incomming ( struct engine *e );
int engine_exchange_rigid ( struct engine *e );
int engine_exchange_rigid_async ( struct engine *e );
int engine_exchange_rigid_async_run ( struct engine *e );
int engine_exchange_rigid_wait ( struct engine *e );
int engine_exchange_wait ( struct engine *e );
#endif

#if defined(HAVE_CUDA) && defined(WITH_CUDA)
int engine_nonbond_cuda ( struct engine *e );
int engine_cuda_load ( struct engine *e );
int engine_cuda_load_parts ( struct engine *e );
int engine_cuda_unload_parts ( struct engine *e );
int engine_cuda_setdevice ( struct engine *e , int id );
int engine_cuda_setdevices ( struct engine *e , int nr_devices , int *ids );
int engine_split_METIS ( struct engine *e, int N, int flags);
#endif

#ifdef WITH_METIS
int engine_split_METIS ( struct engine *e, int N, int flags);
#endif

MDCORE_END_DECLS
#endif // INCLUDE_ENGINE_H_

