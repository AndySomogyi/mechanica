/*******************************************************************************
 * This file is part of mdcore.
 * Coypright (c) 2011 Pedro Gonnet (gonnet@maths.ox.ac.uk)
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

/* Wrappers for ppm calls. */

/* Define some constants (as generated in gfortran). */
#define ppm_kind_double               8
#define ppm_kind_single               4
#define ppm_integer                   13
#define ppm_logical                   17
#define ppm_char                      256
#define ppm_param_map_pop             1
#define ppm_param_map_push            2
#define ppm_param_map_global          3
#define ppm_param_map_send            4
#define ppm_param_map_ghost_get       5
#define ppm_param_map_ghost_put       6
#define ppm_param_map_partial         7
#define ppm_param_map_remap           8
#define ppm_param_map_cancel          9
#define ppm_param_map_init            10
#define ppm_param_connect_distribute  1
#define ppm_param_connect_send        2
#define ppm_param_connect_prune       3
#define ppm_param_pop_replace         1 
#define ppm_param_pop_add             2
#define ppm_param_kernel_laplace2d_p2  1 
#define ppm_param_kernel_laplace3d_p2  2
#define ppm_param_kernel_sph2d_p2      3
#define ppm_param_kernel_dx_sph2d_p2   4
#define ppm_param_kernel_dy_sph2d_p2   5
#define ppm_param_kernel_ddx_sph2d_p2  6
#define ppm_param_kernel_ddy_sph2d_p2  7
#define ppm_param_kernel_dxdy_sph2d_p2 8
#define ppm_param_kernel_fast3d        9
#define ppm_param_kernel_fast3d_dx    10
#define ppm_param_kernel_fast3d_dy    11
#define ppm_param_kernel_fast3d_dz    12
#define ppm_param_kernel_fast3d_lap   13
#define ppm_param_loadbal_sar         1 
#define ppm_param_update_replace      1
#define ppm_param_update_average      2
#define ppm_param_update_expfavg      3
#define ppm_param_decomp_tree         1
#define ppm_param_decomp_pruned_cell  2
#define ppm_param_decomp_bisection    3
#define ppm_param_decomp_xpencil      4
#define ppm_param_decomp_ypencil      5
#define ppm_param_decomp_zpencil      6
#define ppm_param_decomp_cuboid       7
#define ppm_param_decomp_user_defined 8
#define ppm_param_decomp_xy_slab      10
#define ppm_param_decomp_xz_slab      11
#define ppm_param_decomp_yz_slab      12
#define ppm_param_decomp_cartesian    13
#define ppm_param_tree_bin            1
#define ppm_param_tree_quad           2
#define ppm_param_tree_oct            3
#define ppm_param_assign_internal     1
#define ppm_param_assign_nodal_cut    2
#define ppm_param_assign_nodal_comm   3
#define ppm_param_assign_dual_cut     4
#define ppm_param_assign_dual_comm    5
#define ppm_param_assign_user_defined 6
#define ppm_param_assign_ngp          0
#define ppm_param_assign_cic          1
#define ppm_param_assign_tcs          2
#define ppm_param_assign_mp4          3
#define ppm_param_assign_m3p6         4
#define ppm_param_mesh_refine         0
#define ppm_param_mesh_coarsen        1
#define ppm_param_bcdef_periodic      1
#define ppm_param_bcdef_freespace     2
#define ppm_param_bcdef_symmetry      3
#define ppm_param_bcdef_neumann       4
#define ppm_param_bcdef_dirichlet     5
#define ppm_param_bcdef_robin         6
#define ppm_param_bcdef_antisymmetry  7
#define ppm_param_alloc_fit           1
#define ppm_param_alloc_fit_preserve  2
#define ppm_param_alloc_grow          3
#define ppm_param_alloc_grow_preserve 4
#define ppm_param_dealloc             5
#define ppm_param_io_read             1
#define ppm_param_io_write            2
#define ppm_param_io_read_write       3
#define ppm_param_io_replace          4
#define ppm_param_io_append           5
#define ppm_param_io_ascii            6
#define ppm_param_io_binary           7
#define ppm_param_io_distributed      8
#define ppm_param_io_centralized      9
#define ppm_param_io_same             10
#define ppm_param_io_root             11
#define ppm_param_io_sum              12
#define ppm_param_io_split            13
#define ppm_param_io_concat           14
#define ppm_param_io_single           15
#define ppm_param_io_double           16
#define ppm_error_notice              -1
#define ppm_error_warning             -2
#define ppm_error_error               -3
#define ppm_error_fatal               -4
#define ppm_param_topo_part           1
#define ppm_param_topo_field          2
#define ppm_param_id_internal         3
#define ppm_param_id_user             4
#define ppm_param_undefined           -1
#define ppm_param_success             0
#define ppm_param_topo_undefined      -1
#define ppm_param_rmsh_kernel_bsp2    1
#define ppm_param_rmsh_kernel_mp4     2
#define ppm_param_rmsh_kernel_m3p6    3


/* Assumed size/shape array wrapper (from gfortran). */
typedef struct dope_vec_GNU_ {
    void*   base_addr;     /* base address of the array */
    void*   base;          /* base offset */
    size_t  dtype;         /* elem_size, type (3 bits) and rank (3 bits) */

    struct {
        size_t stride_mult;  /* distance between successive elements (elements) */
        size_t lower_bound;  /* first array index for a given dimension */
        size_t upper_bound;  /* last array index for a given dimension */
        } dim[7];
    } dvec;

#define GFC_DTYPE_RANK_MASK 0x07
#define GFC_DTYPE_TYPE_SHIFT 3
#define GFC_DTYPE_TYPE_MASK 0x38
#define GFC_DTYPE_SIZE_SHIFT 6

#define GFC_DESCRIPTOR_RANK(desc) ((desc)->dtype & GFC_DTYPE_RANK_MASK)
#define GFC_DESCRIPTOR_TYPE(desc) (((desc)->dtype & GFC_DTYPE_TYPE_MASK) >> GFC_DTYPE_TYPE_SHIFT)
#define GFC_DESCRIPTOR_SIZE(desc) ((desc)->dtype >> GFC_DTYPE_SIZE_SHIFT)
#define GFC_DESCRIPTOR_DATA(desc) ((desc)->base_addr)
#define GFC_DESCRIPTOR_DTYPE(desc) ((desc)->dtype)

enum {
    GFC_DTYPE_UNKNOWN = 0,
    GFC_DTYPE_INTEGER,
    /* TODO: recognize logical types.  */
    GFC_DTYPE_LOGICAL,
    GFC_DTYPE_REAL,
    GFC_DTYPE_COMPLEX,
    GFC_DTYPE_DERIVED,
    GFC_DTYPE_CHARACTER
    };
    
inline void dvec_fill_1d ( dvec *d , void *data , size_t dtype , int elem_size , int dim ) {
    d->base_addr = data;
    d->dtype = ( elem_size << GFC_DTYPE_SIZE_SHIFT ) |
        ( (dtype << GFC_DTYPE_TYPE_SHIFT) & GFC_DTYPE_TYPE_MASK ) |
        1;
    d->dim[0].stride_mult = 1;
    d->dim[0].lower_bound = 1;
    d->dim[0].upper_bound = dim;
    d->base = (void *)( (size_t)-1 );
    }
inline void dvec_fill_2d ( dvec *d , void *data , size_t dtype , int elem_size , int dim0 , int dim1 ) {
    d->base_addr = data;
    d->dtype = ( elem_size << GFC_DTYPE_SIZE_SHIFT ) |
        ( (dtype << GFC_DTYPE_TYPE_SHIFT) & GFC_DTYPE_TYPE_MASK ) |
        2;
    d->dim[0].stride_mult = 1;
    d->dim[0].lower_bound = 1;
    d->dim[0].upper_bound = dim0;
    d->dim[1].stride_mult = dim0;
    d->dim[1].lower_bound = 1;
    d->dim[1].upper_bound = dim1;
    d->base = (void *)( (size_t)-1 - dim0 );
    }
inline void dvec_fill_3d ( dvec *d , void *data , size_t dtype , int elem_size , int dim0 , int dim1 , int dim2 ) {
    d->base_addr = data;
    d->dtype = ( elem_size << GFC_DTYPE_SIZE_SHIFT ) |
        ( (dtype << GFC_DTYPE_TYPE_SHIFT) & GFC_DTYPE_TYPE_MASK ) |
        2;
    d->dim[0].stride_mult = 1;
    d->dim[0].lower_bound = 1;
    d->dim[0].upper_bound = dim0;
    d->dim[1].stride_mult = dim0;
    d->dim[1].lower_bound = 1;
    d->dim[1].upper_bound = dim1;
    d->dim[2].stride_mult = dim0*dim1;
    d->dim[2].lower_bound = 1;
    d->dim[2].upper_bound = dim2;
    d->base = (void *)( (size_t)-1 - dim0 - dim0*dim1 );
    }
inline void dvec_dump ( dvec *d , FILE *out ) {
    int k, rank = GFC_DESCRIPTOR_RANK(d);
    fprintf( out , "dvec.base_addr = %x\n" , (size_t)(d->base_addr) );
    fprintf( out , "dvec.base = %x\n" , (size_t)(d->base) );
    fprintf( out , "dvec.dtype = %i|%i|%i (s|t|r)\n" , GFC_DESCRIPTOR_SIZE(d) , GFC_DESCRIPTOR_TYPE(d) , rank );
    for ( k = 0 ; k < rank ; k++ ) {
        fprintf( out , "dvec.dim[%i].stride_mult = %i\n" , k , d->dim[k].stride_mult );
        fprintf( out , "dvec.dim[%i].lower_bound = %i\n" , k , d->dim[k].lower_bound );
        fprintf( out , "dvec.dim[%i].upper_bound = %i\n" , k , d->dim[k].upper_bound );
        }
    }


/* External function wrappers. */
extern void __ppm_module_init_MOD_ppm_init (
    int *dim , int *prec , int *tolexp , int *comm , int *debug , int *info , int *log , int *stderr , int *stdout );
extern void __ppm_module_mktopo_MOD_ppm_topo_mkpart_d ( 
    int *topoid , dvec *xp , int *npart , int *decomp , int *assig , dvec *min_phys , dvec *max_phys , dvec *bcdef , double *ghostsize , dvec *cost , int *info , dvec *pcost , dvec *user_minsub , dvec *user_maxsub , int *user_nsubs , dvec *user_sub2proc );
extern void __ppm_module_mktopo_MOD_ppm_topo_mkgeom_d ( 
    int *topoid , int *decomp , int *assig , dvec *min_phys , dvec *max_phys , dvec *bcdef , double *ghostsize , dvec *cost , int *info , dvec *user_minsub , dvec *user_maxsub , int *user_nsubs , dvec *user_sub2proc );
extern void __ppm_module_map_part_global_MOD_ppm_map_part_global_d (
    int *topoid , dvec *xp , int *npart , int *info , dvec *userdef_part2proc );
extern void __ppm_module_map_part_MOD_ppm_map_part_push_1dd (
    dvec *pdata , int *npart , int *info , int *pushpp );
extern void __ppm_module_map_part_MOD_ppm_map_part_push_1di (
    dvec *pdata , int *npart , int *info , int *pushpp );
extern void __ppm_module_map_part_MOD_ppm_map_part_push_2dd (
    dvec *pdata , int *lda , int *npart , int *info , int *pushpp );
extern void __ppm_module_map_part_MOD_ppm_map_part_push_2di (
    dvec *pdata , int *lda , int *npart , int *info , int *pushpp );
extern void __ppm_module_map_part_MOD_ppm_map_part_pop_1dd (
    dvec *pdata , int *npart , int *mpart , int *info );
extern void __ppm_module_map_part_MOD_ppm_map_part_pop_1di (
    dvec *pdata , int *npart , int *mpart , int *info );
extern void __ppm_module_map_part_MOD_ppm_map_part_pop_2dd (
    dvec *pdata , int *lda , int *npart , int *mpart , int *info );
extern void __ppm_module_map_part_MOD_ppm_map_part_pop_2di (
    dvec *pdata , int *lda , int *npart , int *mpart , int *info );
extern void __ppm_module_map_part_MOD_ppm_map_part_send (
    int *npart , int *mpart , int *info );
extern void __ppm_module_topo_get_MOD_ppm_topo_getextent (
    int *topoid , dvec *min_phys , dvec *max_phys , int *info );
extern void __ppm_module_map_part_ghost_MOD_ppm_map_part_ghost_get_d (
    int *topoid , dvec *xp , int *lda , int *npart , int *issym , double *ghostsize , int *info );
extern void __ppm_module_map_part_ghost_MOD_ppm_map_part_ghost_pop_1dd (
    dvec *pdata , int *lda , int *npart , int *mpart , int *info );
extern void __ppm_module_map_part_ghost_MOD_ppm_map_part_ghost_pop_1di (
    dvec *pdata , int *lda , int *npart , int *mpart , int *info );
extern void __ppm_module_map_part_ghost_MOD_ppm_map_part_ghost_pop_2dd (
    dvec *pdata , int *lda , int *npart , int *mpart , int *info );
extern void __ppm_module_map_part_ghost_MOD_ppm_map_part_ghost_pop_2di (
    dvec *pdata , int *lda , int *npart , int *mpart , int *info );
extern void __ppm_module_map_part_partial_MOD_ppm_map_part_partial_d (
    int *topoid , dvec *xp , int *npart , int *info , int *ignore );
extern void __ppm_module_impose_part_bc_MOD_ppm_impose_part_bc_d (
    int *topoid , dvec *xp , int *npart , int *info );
extern void __ppm_module_topo_check_MOD_ppm_topo_check_d (
    int *topoid , dvec *xp , int *npart , int *topo_ok , int *info );
    
    
/* Wrappers to convert params from C to gfortran */
inline void ppm_init ( int dim , int prec , int tolexp , int comm , int debug , int *info , int log , int stderr , int stdout ) {
    __ppm_module_init_MOD_ppm_init( &dim , &prec , &tolexp , &comm , &debug , info , &log , &stderr , &stdout );
    }
inline void ppm_topo_mkgeom ( int *topoid , int decomp , int assig , double *min_phys , double *max_phys , int *bcdef , double ghostsize , double **cost , int *ncost , int *info ) {
    dvec dmin_phys, dmax_phys, dbcdef, dcost;
    dvec_fill_1d( &dmin_phys , min_phys , GFC_DTYPE_REAL , sizeof(double) , 3 );
    dvec_fill_1d( &dmax_phys , max_phys , GFC_DTYPE_REAL , sizeof(double) , 3 );
    dvec_fill_1d( &dbcdef , bcdef , GFC_DTYPE_INTEGER , sizeof(int) , 6 );
    dvec_fill_1d( &dcost , NULL , GFC_DTYPE_REAL , sizeof(double) , 0 );
    __ppm_module_mktopo_MOD_ppm_topo_mkgeom_d( topoid , &decomp , &assig , &dmin_phys , &dmax_phys , &dbcdef , &ghostsize , &dcost , info , NULL , NULL , NULL , NULL );
    *cost = (double *)dcost.base_addr;
    *ncost = dcost.dim[0].upper_bound - dcost.dim[0].lower_bound + 1;
    }
inline void ppm_topo_mkpart ( int *topoid , double *xp , int len , int npart , int decomp , int assig , double *min_phys , double *max_phys , int *bcdef , double ghostsize , double **cost , int *ncost , int *info ) {
    dvec dxp, dmin_phys, dmax_phys, dbcdef, dcost;
    dvec_fill_2d( &dxp , xp , GFC_DTYPE_REAL , sizeof(double) , 3 , len );
    dvec_fill_1d( &dmin_phys , min_phys , GFC_DTYPE_REAL , sizeof(double) , 3 );
    dvec_fill_1d( &dmax_phys , max_phys , GFC_DTYPE_REAL , sizeof(double) , 3 );
    dvec_fill_1d( &dbcdef , bcdef , GFC_DTYPE_INTEGER , sizeof(int) , 6 );
    dvec_fill_1d( &dcost , NULL , GFC_DTYPE_REAL , sizeof(double) , 0 );
    __ppm_module_mktopo_MOD_ppm_topo_mkpart_d( topoid , &dxp , &npart , &decomp , &assig , &dmin_phys , &dmax_phys , &dbcdef , &ghostsize , &dcost , info , NULL , NULL , NULL , NULL , NULL );
    *cost = (double *)dcost.base_addr;
    *ncost = dcost.dim[0].upper_bound - dcost.dim[0].lower_bound + 1;
    }
inline void ppm_map_part_global ( int topoid , double *xp , int len , int npart , int *info ) {
    dvec dxp;
    dvec_fill_2d( &dxp , xp , GFC_DTYPE_REAL , sizeof(double) , 3 , len );
    __ppm_module_map_part_global_MOD_ppm_map_part_global_d( &topoid , &dxp , &npart , info , NULL );
    }
inline void ppm_map_part_push_1dd ( double *pdata , int len , int npart , int *info , int pushpp ) {
    dvec dpdata;
    dvec_fill_1d( &dpdata , pdata , GFC_DTYPE_REAL , sizeof(double) , len );
    __ppm_module_map_part_MOD_ppm_map_part_push_1dd( &dpdata , &npart , info , &pushpp );
    }
inline void ppm_map_part_push_1di ( int *pdata , int len , int npart , int *info , int pushpp ) {
    dvec dpdata;
    dvec_fill_1d( &dpdata , pdata , GFC_DTYPE_INTEGER , sizeof(int) , len );
    __ppm_module_map_part_MOD_ppm_map_part_push_1di( &dpdata , &npart , info , &pushpp );
    }
inline void ppm_map_part_push_2dd ( double *pdata , int lda , int len , int npart , int *info , int pushpp ) {
    dvec dpdata;
    dvec_fill_2d( &dpdata , pdata , GFC_DTYPE_REAL , sizeof(double) , lda , len );
    __ppm_module_map_part_MOD_ppm_map_part_push_2dd( &dpdata , &lda , &npart , info , &pushpp );
    }
inline void ppm_map_part_push_2di ( int *pdata , int lda , int len , int npart , int *info , int pushpp ) {
    dvec dpdata;
    dvec_fill_2d( &dpdata , pdata , GFC_DTYPE_INTEGER , sizeof(int) , lda , len );
    __ppm_module_map_part_MOD_ppm_map_part_push_2di( &dpdata , &lda , &npart , info , &pushpp );
    }
inline void ppm_map_part_send ( int *npart , int *mpart , int *info ) {
    __ppm_module_map_part_MOD_ppm_map_part_send( npart , mpart , info );
    }
inline void ppm_map_part_pop_1dd ( double **pdata , int *len , int *npart , int *mpart , int *info ) {
    dvec dpdata;
    dvec_fill_1d( &dpdata , *pdata , GFC_DTYPE_REAL , sizeof(double) , *len );
    __ppm_module_map_part_MOD_ppm_map_part_pop_1dd( &dpdata , npart , mpart , info );
    *pdata = (double *)dpdata.base_addr;
    *len = dpdata.dim[0].upper_bound - dpdata.dim[0].lower_bound + 1;
    }
inline void ppm_map_part_pop_1di ( int **pdata , int *len , int *npart , int *mpart , int *info ) {
    dvec dpdata;
    dvec_fill_1d( &dpdata , *pdata , GFC_DTYPE_INTEGER , sizeof(int) , *len );
    __ppm_module_map_part_MOD_ppm_map_part_pop_1di( &dpdata , npart , mpart , info );
    *pdata = (int *)dpdata.base_addr;
    *len = dpdata.dim[0].upper_bound - dpdata.dim[0].lower_bound + 1;
    }
inline void ppm_map_part_pop_2dd ( double **pdata , int lda , int *len , int *npart , int *mpart , int *info ) {
    dvec dpdata;
    dvec_fill_2d( &dpdata , *pdata , GFC_DTYPE_REAL , sizeof(double) , lda , *len );
    __ppm_module_map_part_MOD_ppm_map_part_pop_2dd( &dpdata , &lda , npart , mpart , info );
    *pdata = (double *)dpdata.base_addr;
    *len = dpdata.dim[1].upper_bound - dpdata.dim[1].lower_bound + 1;
    }
inline void ppm_map_part_pop_2di ( int **pdata , int *len , int lda , int *npart , int *mpart , int *info ) {
    dvec dpdata;
    dvec_fill_2d( &dpdata , *pdata , GFC_DTYPE_INTEGER , sizeof(int) , lda , *len );
    __ppm_module_map_part_MOD_ppm_map_part_pop_2di( &dpdata , &lda , npart , mpart , info );
    *pdata = (int *)dpdata.base_addr;
    *len = dpdata.dim[1].upper_bound - dpdata.dim[1].lower_bound + 1;
    }
inline void ppm_topo_getextent ( int topoid , double *min_phys , double *max_phys , int *info ) {
    dvec dmin_phys, dmax_phys;
    dvec_fill_1d( &dmin_phys , min_phys , GFC_DTYPE_REAL , sizeof(double) , 3 );
    dvec_fill_1d( &dmax_phys , max_phys , GFC_DTYPE_REAL , sizeof(double) , 3 );
    __ppm_module_topo_get_MOD_ppm_topo_getextent ( &topoid , &dmin_phys , &dmax_phys, info );
    }
inline void ppm_map_part_ghost_get ( int topoid , double *xp , int lda , int len , int npart , int issym , double ghostsize , int *info ) {
    dvec dxp;
    dvec_fill_2d( &dxp , xp , GFC_DTYPE_REAL , sizeof(double) , 3 , len );
    __ppm_module_map_part_ghost_MOD_ppm_map_part_ghost_get_d( &topoid , &dxp , &lda , &npart , &issym , &ghostsize , info );
    }
inline void ppm_map_part_ghost_pop_1dd ( double **pdata , int *len , int npart , int *mpart , int *info ) {
    dvec dpdata;
    int lda = 1;
    dvec_fill_1d( &dpdata , *pdata , GFC_DTYPE_REAL , sizeof(double) , *len );
    __ppm_module_map_part_ghost_MOD_ppm_map_part_ghost_pop_1dd( &dpdata , &lda , &npart , mpart , info );
    *pdata = (double *)dpdata.base_addr;
    *len = dpdata.dim[0].upper_bound - dpdata.dim[0].lower_bound + 1;
    }
inline void ppm_map_part_ghost_pop_1di ( int **pdata , int *len , int npart , int *mpart , int *info ) {
    dvec dpdata;
    int lda = 1;
    dvec_fill_1d( &dpdata , *pdata , GFC_DTYPE_INTEGER , sizeof(int) , *len );
    __ppm_module_map_part_ghost_MOD_ppm_map_part_ghost_pop_1di( &dpdata , &lda , &npart , mpart , info );
    *pdata = (int *)dpdata.base_addr;
    *len = dpdata.dim[0].upper_bound - dpdata.dim[0].lower_bound + 1;
    }
inline void ppm_map_part_ghost_pop_2dd ( double **pdata , int lda , int *len , int npart , int *mpart , int *info ) {
    dvec dpdata;
    dvec_fill_2d( &dpdata , *pdata , GFC_DTYPE_REAL , sizeof(double) , lda , *len );
    __ppm_module_map_part_ghost_MOD_ppm_map_part_ghost_pop_2dd( &dpdata , &lda , &npart , mpart , info );
    *pdata = (double *)dpdata.base_addr;
    *len = dpdata.dim[1].upper_bound - dpdata.dim[1].lower_bound + 1;
    }
inline void ppm_map_part_ghost_pop_2di ( int **pdata , int *len , int lda , int npart , int *mpart , int *info ) {
    dvec dpdata;
    dvec_fill_2d( &dpdata , *pdata , GFC_DTYPE_INTEGER , sizeof(int) , lda , *len );
    __ppm_module_map_part_ghost_MOD_ppm_map_part_ghost_pop_2di( &dpdata , &lda , &npart , mpart , info );
    *pdata = (int *)dpdata.base_addr;
    *len = dpdata.dim[1].upper_bound - dpdata.dim[1].lower_bound + 1;
    }
inline void ppm_map_part_partial ( int topoid , double *xp , int len , int npart , int *info ) {
    dvec dxp;
    dvec_fill_2d( &dxp , xp , GFC_DTYPE_REAL , sizeof(double) , 3 , len );
    __ppm_module_map_part_partial_MOD_ppm_map_part_partial_d( &topoid , &dxp , &npart , info , NULL );
    }
inline void ppm_impose_part_bc ( int topoid , double *xp , int len , int npart , int *info ) {
    dvec dxp;
    dvec_fill_2d( &dxp , xp , GFC_DTYPE_REAL , sizeof(double) , 3 , len );
    __ppm_module_impose_part_bc_MOD_ppm_impose_part_bc_d( &topoid , &dxp , &npart , info );
    }
inline void ppm_topo_check ( int topoid , double *xp , int len , int npart , int *topo_ok , int *info ) {
    dvec dxp;
    dvec_fill_2d( &dxp , xp , GFC_DTYPE_REAL , sizeof(double) , 3 , len );
    __ppm_module_topo_check_MOD_ppm_topo_check_d( &topoid , &dxp , &npart , topo_ok , info );
    }
    



