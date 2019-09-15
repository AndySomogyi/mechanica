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

/* Include some standard headers */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include "cycle.h"

/* MPI headers. */
#include <mpi.h>

/* Include mdcore. */
#include "mdcore.h"


/* Wrappers for ppm calls. */
#include "ppm.h"

/* Ticks Per Second. */
#ifndef CPU_TPS
    #define CPU_TPS 2.67e+9
#endif

/* Engine flags? */
#ifndef ENGINE_FLAGS
    #define ENGINE_FLAGS engine_flag_tuples
#endif

/* Enumeration for the different timers */
enum {
    tid_step = 0,
    tid_shake,
    tid_gexch0,
    tid_gexch1,
    tid_pexch0,
    tid_pexch1,
    tid_load,
    tid_uload,
    tid_resolv,
    tid_temp
    };


/* The main routine -- this is where it all happens. */

int main ( int argc , char *argv[] ) {


    /* Simulation constants. */
    double origin[3] = { 0.0 , 0.0 , 0.0 };
    // double dim[3] = { 16.0 , 16.0 , 16.0 };
    // int nr_mols = 129024, nr_parts = nr_mols*3;
    double dim[3] = { 8.0 , 8.0 , 8.0 };
    int nr_mols = 16128, nr_parts = nr_mols*3;
    double cutoff = 1.0;


    /* Local variables. */
    int res = 0, myrank;
    double *xp = NULL, *vp = NULL, x[3], v[3];
    int *pid = NULL, *vid = NULL, *ptype = NULL;
    int step, i, j, k, nx, ny, nz, id, cid;
    double hx, hy, hz, temp;
    double vtot[3] = { 0.0 , 0.0 , 0.0 };
    FILE *dump;
    char fname[100];
    struct {
        struct particle *p;
        struct unit_cell *c;
        } globloc[nr_parts];
    double old_O[3], old_H1[3], old_H2[3], new_O[3], new_H1[3], new_H2[3];
    double v_OH1[3], v_OH2[3], v_HH[3], ldim[3], vp_O[3], vp_H1[3], vp_H2[3];
    double d_OH1, d_OH2, d_HH, lambda;
    double vcom_tot[6], vcom_tot_x, vcom_tot_y, vcom_tot_z, ekin, epot, vcom[3], w, v2;
    ticks tic, toc, tic_step, toc_step, timers[10];
    double itpms = 1000.0 / CPU_TPS;
    struct particle *p_O, *p_H1, *p_H2, *p;
    struct unit_cell *c_O, *c_H1, *c_H2;
    int verbose = 0;
    
    
    /* PPM variables. */
    int ppm_debug = 0;
    int topoid = 0;
    int bc[6] = { ppm_param_bcdef_periodic , ppm_param_bcdef_periodic , ppm_param_bcdef_periodic , ppm_param_bcdef_periodic , ppm_param_bcdef_periodic , ppm_param_bcdef_periodic };
    double *cost = NULL;
    int ncost, ppm_npart = 0, ppm_mpart = 0;
    double ppm_minphys[3], ppm_maxphys[3], ppm_dim[3];
    int xp_len = 0, vp_len = 0, pid_len = 0, vid_len = 0;

    
    /* mdcore stuff. */
    struct engine e;
    struct potential *pot_OO, *pot_OH, *pot_HH;
    int nr_runners = 1, nr_steps = 1000, nr_nodes;
    
    
    /* Start the clock. */
    tic = getticks();
    
    
    /* Start by initializing MPI. */
    if ( ( res = MPI_Init( &argc , &argv ) ) != MPI_SUCCESS ) {
        printf( "main: call to MPI_Init failed with error %i.\n" , res );
        return -1;
        }
    if ( ( res = MPI_Comm_rank( MPI_COMM_WORLD , &myrank ) ) != MPI_SUCCESS ) {
        printf( "main: call to MPI_Comm_rank failed with error %i.\n" , res );
        return -1;
        }
    if ( myrank == 0 ) {
        printf( "main[%i]: MPI is up and running...\n" , myrank );
        fflush(stdout);
        }
    if ( ( res = MPI_Comm_size( MPI_COMM_WORLD , &nr_nodes ) != MPI_SUCCESS ) ) {
        printf("main[%i]: MPI_Comm_size failed with error %i.\n",myrank,res);
        errs_dump(stdout);
        return -1;
        }
    
    
    /* Initialize our own input parameters. */
    if ( argc > 1 )
        nr_runners = atoi( argv[1] );
    if ( argc > 2 )
        nr_steps = atoi( argv[2] );
        
    
    /* Now try calling ppm_init. */
    ppm_init( 3 , ppm_kind_double , -15 , MPI_Comm_c2f(MPI_COMM_WORLD) , ppm_debug , &res , 0 , 0 , 0 );
    if ( res != 0 ) {
        printf( "main[%i]: call to ppm_init failed with error %i.\n" , myrank , res );
        return -1;
        }
    if ( myrank == 0 ) {
        printf( "main[%i]: PPM is up and running...\n" , myrank ); fflush(stdout);
        }
        
        
    /* Make the topology. */
    ppm_topo_mkgeom( &topoid , ppm_param_decomp_bisection , ppm_param_assign_internal , origin , dim , bc , cutoff , &cost , &ncost , &res );
    /* ppm_topo_mkpart( &topoid , xp , ppm_npart , ppm_param_decomp_cuboid , ppm_param_assign_internal , origin , dim , bc , cutoff , &cost , &ncost , &res ); */
    if ( res != 0 ) {
        printf( "main[%i]: call to ppm_mktopo failed with error %i.\n" , myrank , res );
        return -1;
        }
    if ( myrank == 0 ) {
        printf( "main[%i]: Created topology (ncost=%i).\n" , myrank , ncost ); fflush(stdout);
        }
        
        
    /* Generate the particles for this simulation. */
    if ( myrank == 0 ) {
        srand(6178);
        if ( ( xp = (double *)malloc( sizeof(double) * nr_parts * 3 ) ) == NULL ||
             ( vp = (double *)malloc( sizeof(double) * nr_parts * 3 ) ) == NULL ||
             ( pid = (int *)malloc( sizeof(int) * nr_parts ) ) == NULL ||
             ( vid = (int *)malloc( sizeof(int) * nr_parts ) ) == NULL ) {
             printf( "main[%i]: allocation of particle data failed!\n" , myrank );
             return -1;
             }
        xp_len = nr_parts; vp_len = nr_parts; pid_len = nr_parts; vid_len = nr_parts;
        printf("main[%i]: initializing particles... " , myrank); fflush(stdout);
        nx = ceil( pow( nr_mols , 1.0/3 ) ); hx = dim[0] / nx;
        ny = ceil( sqrt( ((double)nr_mols) / nx ) ); hy = dim[1] / ny;
        nz = ceil( ((double)nr_mols) / nx / ny ); hz = dim[2] / nz;
        for ( i = 0 ; i < nx ; i++ ) {
            x[0] = 0.05 + i * hx;
            for ( j = 0 ; j < ny ; j++ ) {
                x[1] = 0.05 + j * hy;
                for ( k = 0 ; k < nz && k + nz * ( j + ny * i ) < nr_mols ; k++ ) {
                    id = 3 * (k + nz * ( j + ny * i ));
                    x[2] = 0.5 + k * hz;
                    v[0] = ((double)rand()) / RAND_MAX - 0.5;
                    v[1] = ((double)rand()) / RAND_MAX - 0.5;
                    v[2] = ((double)rand()) / RAND_MAX - 0.5;
                    temp = 0.675 / sqrt( v[0]*v[0] + v[1]*v[1] + v[2]*v[2] );
                    v[0] *= temp; v[1] *= temp; v[2] *= temp;
                    vtot[0] += v[0]; vtot[1] += v[1]; vtot[2] += v[2];
                    /* Add oxygen. */
                    xp[ 3*id + 0 ] = x[0]; vp[ 3*id + 0 ] = v[0];
                    xp[ 3*id + 1 ] = x[1]; vp[ 3*id + 1 ] = v[1];
                    xp[ 3*id + 2 ] = x[2]; vp[ 3*id + 2 ] = v[2];
                    pid[ id ] = id;
                    vid[ id ] = k + nz * ( j + ny * i );
                    x[0] += 0.1;
                    id += 1;
                    /* Add first hydrogen atom. */
                    xp[ 3*id + 0 ] = x[0]; vp[ 3*id + 0 ] = v[0];
                    xp[ 3*id + 1 ] = x[1]; vp[ 3*id + 1 ] = v[1];
                    xp[ 3*id + 2 ] = x[2]; vp[ 3*id + 2 ] = v[2];
                    pid[ id ] = id;
                    vid[ id ] = k + nz * ( j + ny * i );
                    x[0] -= 0.13333;
                    x[1] += 0.09428;
                    id += 1;
                    /* Add second hydrogen atom. */
                    xp[ 3*id + 0 ] = x[0]; vp[ 3*id + 0 ] = v[0];
                    xp[ 3*id + 1 ] = x[1]; vp[ 3*id + 1 ] = v[1];
                    xp[ 3*id + 2 ] = x[2]; vp[ 3*id + 2 ] = v[2];
                    pid[ id ] = id;
                    vid[ id ] = k + nz * ( j + ny * i );
                    x[0] += 0.03333;
                    x[1] -= 0.09428;
                    }
                }
            }
        for ( i = 0 ; i < nr_parts ; i++ )
            for ( k = 0 ; k < 3 ; k++ )
                vp[ 3*i + k ] -= vtot[k] / nr_mols;
        printf("done.\n"); fflush(stdout);
        printf("main[%i]: generated %i particles.\n", myrank, nr_parts);
        ppm_npart = nr_parts; ppm_mpart = ppm_npart;
        /* dump = fopen("parts_000.dump","w");
        for ( i = 0 ; i < nr_parts ; i++ )
            fprintf(dump,"%e %e %e\n",xp[3*i+0],xp[3*i+1],xp[3*i+2]);
        fclose(dump);*/
        }
    if ( ( res = MPI_Barrier( MPI_COMM_WORLD ) ) != MPI_SUCCESS ) {
        printf( "main[%i]: call to MPI_Barrier failed with error %i.\n" , myrank , res );
        return -1;
        }
    
    
    /* Distribute the particle data over all processors. */
    ppm_impose_part_bc( topoid , xp , xp_len , ppm_npart , &res );
    if ( res != 0 ) {
        printf( "main[%i]: call to ppm_impose_part_bc failed with error %i.\n" , myrank , res );
        return -1;
        }
    ppm_map_part_global( topoid , xp , xp_len , ppm_npart , &res );
    if ( res != 0 ) {
        printf( "main[%i]: call to ppm_map_part_global failed with error %i.\n" , myrank , res );
        return -1;
        }
    ppm_map_part_push_2dd( vp , 3 , vp_len , ppm_npart , &res , 0 );
    if ( res != 0 ) {
        printf( "main[%i]: call to ppm_map_part_push_2dd failed with error %i.\n" , myrank , res );
        return -1;
        }
    ppm_map_part_push_1di( pid , pid_len , ppm_npart , &res , 0 );
    if ( res != 0 ) {
        printf( "main[%i]: call to ppm_map_part_push_1di failed with error %i.\n" , myrank , res );
        return -1;
        }
    ppm_map_part_push_1di( vid , vid_len , ppm_npart , &res , 0 );
    if ( res != 0 ) {
        printf( "main[%i]: call to ppm_map_part_push_1di failed with error %i.\n" , myrank , res );
        return -1;
        }
    ppm_map_part_send( &ppm_npart , &ppm_mpart , &res );
    if ( res != 0 ) {
        printf( "main[%i]: call to ppm_map_part_send failed with error %i.\n" , myrank , res );
        return -1;
        }
        
    /* Get the particle data back. */
    ppm_map_part_pop_1di( &vid , &vid_len , &ppm_npart , &ppm_mpart , &res );
    if ( res != 0 ) {
        printf( "main[%i]: call to ppm_map_part_pop_1di failed with error %i.\n" , myrank , res );
        return -1;
        }
    ppm_map_part_pop_1di( &pid , &pid_len , &ppm_npart , &ppm_mpart , &res );
    if ( res != 0 ) {
        printf( "main[%i]: call to ppm_map_part_pop_1di failed with error %i.\n" , myrank , res );
        return -1;
        }
    ppm_map_part_pop_2dd( &vp , 3 , &vp_len , &ppm_npart , &ppm_mpart , &res );
    if ( res != 0 ) {
        printf( "main[%i]: call to ppm_map_part_pop_2dd failed with error %i.\n" , myrank , res );
        return -1;
        }
    ppm_map_part_pop_2dd( &xp , 3 , &xp_len , &ppm_npart , &ppm_mpart , &res );
    if ( res != 0 ) {
        printf( "main[%i]: call to ppm_map_part_pop_2dd failed with error %i.\n" , myrank , res );
        return -1;
        }
    ppm_npart = ppm_mpart;
    printf( "main[%i]: received %i particles. \n" , myrank , ppm_npart );
    
    
    /* Dump the particles. */
    /* sprintf(fname,"parts_%03i.dump",myrank);
    dump = fopen(fname,"w");
    for ( i = 0 ; i < ppm_mpart ; i++ )
        fprintf(dump,"%e %e %e\n",xp[3*i+0],xp[3*i+1],xp[3*i+2]);
    fclose(dump); */
    
    
    /* Get the extent of the domain on this Processor. */
    ppm_topo_getextent( topoid , ppm_minphys , ppm_maxphys , &res );
    if ( res != 0 ) {
        printf( "main[%i]: call to ppm_topo_getextent failed with error %i.\n" , myrank , res );
        return -1;
        }
    /* printf( "main[%i]: ppm_minphys is [ %e , %e , %e ].\n" , myrank , ppm_minphys[0] , ppm_minphys[1] , ppm_minphys[2] );
    printf( "main[%i]: ppm_maxphys is [ %e , %e , %e ].\n" , myrank , ppm_maxphys[0] , ppm_maxphys[1] , ppm_maxphys[2] ); */
    
    /* Extend the physical domain by the cutoff in all dimensions to allow for
       the ghost layers. */
    for ( k = 0 ; k < 3 ; k++ ) {
        ldim[k] = ppm_maxphys[k] - ppm_minphys[k];
        ppm_minphys[k] -= cutoff;
        ppm_maxphys[k] += cutoff;
        ppm_dim[k] = ppm_maxphys[k] - ppm_minphys[k];
        }
    
        
    /* Initialize the engine. */
    printf( "main[%i]: initializing the engine... " , myrank ); fflush(stdout);
    if ( engine_init( &e , ppm_minphys , ppm_dim , cutoff , cutoff , space_periodic_ghost_full , 2 , ENGINE_FLAGS ) != 0 ) {
        printf( "main[%i]: engine_init failed with engine_err=%i.\n" , myrank , engine_err );
        errs_dump(stdout);
        return -1;
        }
    e.dt = 0.002;
    e.time = 0;
    printf("done.\n");
    if ( myrank == 0 )
        printf( "main[%i]: space has %i pairs and %i tuples.\n" , myrank , e.s.nr_pairs , e.s.nr_tuples );
    fflush(stdout);
    
    /* Mark the nodes with potential ghosts. */
    for ( cid = 0 ; cid < e.s.nr_cells ; cid++ )
        if ( e.s.cells[cid].loc[0] == 1 || e.s.cells[cid].loc[0] == e.s.cdim[0]-2 ||
             e.s.cells[cid].loc[1] == 1 || e.s.cells[cid].loc[1] == e.s.cdim[1]-2 ||
             e.s.cells[cid].loc[2] == 1 || e.s.cells[cid].loc[2] == e.s.cdim[2]-2 )
            e.s.cells[cid].flags |= cell_flag_marked;
    /* for ( cid = 0 ; cid < e.s.nr_cells ; cid++ )
        if ( e.s.cells[cid].loc[0] > 0 && e.s.cells[cid].loc[0] < e.s.cdim[0]-1 &&
             e.s.cells[cid].loc[1] > 0 && e.s.cells[cid].loc[1] < e.s.cdim[1]-1 &&
             e.s.cells[cid].loc[2] > 0 && e.s.cells[cid].loc[2] < e.s.cdim[2]-1 )
            e.s.cells[cid].flags |= cell_flag_marked; */
    
    
    /* Register the particle types. */
    if ( engine_addtype( &e , 15.9994 , -0.8476 , "O" , NULL ) < 0 ||
         engine_addtype( &e , 1.00794 , 0.4238 , "H" , NULL ) < 0 ) {
        printf("main[%i]: call to engine_addtype failed.\n",myrank);
        errs_dump(stdout);
        return -1;
        }
        
    /* Initialize the O-H potential. */
    if ( ( pot_OH = potential_create_Ewald( 0.1 , 1.0 , -0.35921288 , 3.0 , 1.0e-4 ) ) == NULL ) {
        printf("main[%i]: potential_create_Ewald failed with potential_err=%i.\n",myrank,potential_err);
        errs_dump(stdout);
        return -1;
        }
    if ( myrank == 0 ) {
        printf("main[%i]: constructed OH-potential with %i intervals.\n",myrank,pot_OH->n); fflush(stdout);
        }

    /* Initialize the H-H potential. */
    if ( ( pot_HH = potential_create_Ewald( 0.1 , 1.0 , 1.7960644e-1 , 3.0 , 1.0e-4 ) ) == NULL ) {
        printf("main[%i]: potential_create_Ewald failed with potential_err=%i.\n",myrank,potential_err);
        errs_dump(stdout);
        return -1;
        }
    if ( myrank == 0 ) {
        printf("main[%i]: constructed HH-potential with %i intervals.\n",myrank,pot_HH->n); fflush(stdout);
        }

    /* Initialize the O-O potential. */
    if ( ( pot_OO = potential_create_LJ126_Ewald( 0.25 , 1.0 , 2.637775819766153e-06 , 2.619222661792581e-03 , 7.1842576e-01 , 3.0 , 1.0e-4 ) ) == NULL ) {
        printf("main[%i]: potential_create_LJ126_Ewald failed with potential_err=%i.\n",myrank,potential_err);
        errs_dump(stdout);
        return -1;
        }
    if ( myrank == 0 ) {
        printf("main[%i]: constructed OO-potential with %i intervals.\n",myrank,pot_OO->n); fflush(stdout);
        }
    
    /* Register these potentials. */
    if ( engine_addpot( &e , pot_OO , 0 , 0 ) < 0 ||
         engine_addpot( &e , pot_HH , 1 , 1 ) < 0 ||
         engine_addpot( &e , pot_OH , 0 , 1 ) < 0 ) {
        printf("main[%i]: call to engine_addpot failed.\n",myrank);
        errs_dump(stdout);
        return -1;
        }
        
    /* Load the engine with the initial set of particles. */
    free( ptype );
    if ( ( ptype = (int *)malloc( sizeof(double) * ppm_npart ) ) == NULL ) {
        printf("main[%i]: failed to re-allocate ptype.\n",myrank);
        return -1;
        }
    for ( k = 0 ; k < ppm_npart ; k++ )
        ptype[k] = ( pid[k] % 3 != 0 );
    if ( ( res = engine_load( &e , xp , vp , ptype , pid , vid , NULL , NULL , ppm_npart ) ) < 0 ) {
        printf("main[%i]: engine_load failed with engine_err=%i.\n",myrank,engine_err);
        errs_dump(stdout);
        return -1;
        }
        
        
    /* Start the engine. */
    if ( engine_start( &e , nr_runners ) != 0 ) {
        printf("main[%i]: engine_start failed with engine_err=%i.\n",myrank,engine_err);
        errs_dump(stdout);
        return -1;
        }
        
        
    /* Timing. */    
    toc = getticks();
    if ( myrank == 0 ) {
        printf("main[%i]: setup took %.3f ms.\n",myrank,(double)(toc-tic) * itpms);
        printf("# step e_pot e_kin temp swaps stalls ms_tot ms_step ms_shake ms_gexch0 ms_gexch1 ms_pexch0 ms_pexch1 ms_load ms_uload ms_resolv ms_temp\n");
        fflush(stdout);
        }
        

    /* Main time-stepping loop. */
    for ( step = 0 ; step < nr_steps ; step++ ) {
    
        /* Start the clock. */
        tic_step = getticks();
        
        /* Start by clearing out the ghost cells. */
        tic = getticks();
        if ( engine_flush_ghosts( &e ) < 0 ) {
            printf("main[%i]: engine_flush_ghosts failed with engine_err=%i.\n",myrank,engine_err);
            errs_dump(stdout);
            return -1;
            }
        /* Pack potential ghost particles into xp and pid. */
        if ( ( ppm_npart = engine_unload_marked( &e , xp , NULL , NULL , pid , vid , NULL , NULL , NULL , xp_len ) ) < 0 ) {
            printf("main[%i]: engine_unload_marked failed with engine_err=%i.\n",myrank,engine_err);
            errs_dump(stdout);
            return -1;
            }
        ppm_mpart = ppm_npart;
        timers[tid_uload] = getticks() - tic;
        if ( verbose && myrank == 0 ) {
            printf( "main[%i]: unloading potential ghosts (%i) took %.3f ms.\n",myrank,ppm_npart,(double)timers[tid_uload] * itpms); fflush(stdout);
            }
    
        /* Get ghost data. */
        tic = getticks();
        ppm_map_part_ghost_get( topoid , xp , 3 , xp_len , ppm_npart , 0 , cutoff , &res );
        if ( res != 0 ) {
            printf( "main[%i]: call to ppm_map_part_ghost_get failed with error %i.\n" , myrank , res );
            return -1;
            }
        ppm_map_part_push_1di( pid , pid_len , ppm_npart , &res , 0 );
        if ( res != 0 ) {
            printf( "main[%i]: call to ppm_map_part_push_1di failed with error %i.\n" , myrank , res );
            return -1;
            }
        ppm_map_part_push_1di( vid , vid_len , ppm_npart , &res , 0 );
        if ( res != 0 ) {
            printf( "main[%i]: call to ppm_map_part_push_1di failed with error %i.\n" , myrank , res );
            return -1;
            }
        ppm_map_part_send( &ppm_npart , &ppm_mpart , &res );
        if ( res != 0 ) {
            printf( "main[%i]: call to ppm_map_part_send failed with error %i.\n" , myrank , res );
            return -1;
            }
        ppm_map_part_pop_1di( &vid , &vid_len , &ppm_npart , &ppm_mpart , &res );
        if ( res != 0 ) {
            printf( "main[%i]: call to ppm_map_part_pop_1di failed with error %i.\n" , myrank , res );
            return -1;
            }
        ppm_map_part_pop_1di( &pid , &pid_len , &ppm_npart , &ppm_mpart , &res );
        if ( res != 0 ) {
            printf( "main[%i]: call to ppm_map_part_pop_1di failed with error %i.\n" , myrank , res );
            return -1;
            }
        ppm_map_part_pop_2dd( &xp , 3 , &xp_len , &ppm_npart , &ppm_mpart , &res );
        if ( res != 0 ) {
            printf( "main[%i]: call to ppm_map_part_pop_2dd failed with error %i.\n" , myrank , res );
            return -1;
            }
        timers[tid_gexch0] = getticks() - tic;
        if ( verbose && myrank == 0 ) {
            printf( "main[%i]: ghost exchange took %.3f ms.\n",myrank,(double)timers[tid_gexch0] * itpms); fflush(stdout);
            }
        if ( 0 && verbose ) {
            printf( "main[%i]: now have %i parts and %i ghosts.\n" , myrank , ppm_npart , ppm_mpart-ppm_npart );
            fflush(stdout);
            }
            
            
        /* Dump the particles. */
        /* sprintf(fname,"parts_%03i.dump",myrank);
        dump = fopen(fname,"w");
        for ( i = 0 ; i < ppm_mpart ; i++ )
            fprintf(dump,"%e %e %e\n",xp[3*i+0],xp[3*i+1],xp[3*i+2]);
        fclose(dump); */


        /* Load the ghosts onto the engine. */
        tic = getticks();
        free( ptype );
        if ( ( ptype = (int *)malloc( sizeof(int) * ppm_mpart ) ) == NULL ) {
            printf("main[%i]: failed to re-allocate ptype.\n",myrank);
            return -1;
            }
        for ( k = ppm_npart ; k < ppm_mpart ; k++ )
            ptype[k] = ( pid[k] % 3 != 0 );
        if ( ( res = engine_load_ghosts( &e , &xp[3*ppm_npart] , NULL , &ptype[ppm_npart] , &pid[ppm_npart] , &vid[ppm_npart] , NULL , NULL , ppm_mpart-ppm_npart ) ) < 0 ) {
            printf("main[%i]: engine_load failed with engine_err=%i.\n",myrank,engine_err);
            errs_dump(stdout);
            return -1;
            }
        timers[tid_load] = getticks() - tic;
        if ( verbose && myrank == 0 ) {
            printf("main[%i]: loading ghosts (%i) took %.3f ms.\n",myrank,ppm_mpart-ppm_npart,(double)timers[tid_load] * itpms); fflush(stdout);
            }
            
            
        /* Compute a step. */
        tic = getticks();
        if ( engine_step( &e ) != 0 ) {
            printf("main: engine_step failed with engine_err=%i.\n",engine_err);
            errs_dump(stdout);
            return -1;
            }
        timers[tid_step] = getticks() - tic;
        if ( verbose && myrank == 0 ) {
            printf("main[%i]: engine_step took %.3f ms.\n",myrank,(double)timers[tid_step] * itpms); fflush(stdout);
            }
            
            
        /* Unload any stray particles. */
        tic = getticks();
        if ( ( ppm_npart = engine_unload_strays( &e , xp , vp , NULL , pid , vid , NULL , NULL , NULL , xp_len ) ) < 0 ) {
            printf("main[%i]: engine_unload_strays failed with engine_err=%i.\n",myrank,engine_err);
            errs_dump(stdout);
            return -1;
            }
        ppm_mpart = ppm_npart;
        timers[tid_uload] += getticks() - tic;
        if ( verbose && myrank == 0 ) {
            printf("main[%i]: unloading strays (%i) took %.3f ms.\n",myrank,ppm_npart,(double)timers[tid_uload] * itpms); fflush(stdout);
            }
        
        
        /* Dump the particles. */
        /* sprintf(fname,"parts_%03i.dump",myrank);
        dump = fopen(fname,"w");
        for ( k = 0 ; k < ppm_mpart ; k++ )
            fprintf(dump,"%e %e %e\n",xp[3*k+0],xp[3*k+1],xp[3*k+2]);
        fclose(dump); */
        
        
        /* Re-distribute the stray particles to the processors. */
        tic = getticks();
        ppm_impose_part_bc( topoid , xp , xp_len , ppm_npart , &res );
        if ( res != 0 ) {
            printf( "main[%i]: call to ppm_impose_part_bc failed with error %i.\n" , myrank , res );
            return -1;
            }
        ppm_map_part_partial( topoid , xp , xp_len , ppm_npart , &res );
        if ( res != 0 ) {
            printf( "main[%i]: call to ppm_map_part_partial failed with error %i.\n" , myrank , res );
            return -1;
            }
        ppm_map_part_push_2dd( vp , 3 , vp_len , ppm_npart , &res , 0 );
        if ( res != 0 ) {
            printf( "main[%i]: call to ppm_map_part_push_2dd failed with error %i.\n" , myrank , res );
            return -1;
            }
        ppm_map_part_push_1di( pid , pid_len , ppm_npart , &res , 0 );
        if ( res != 0 ) {
            printf( "main[%i]: call to ppm_map_part_push_1di failed with error %i.\n" , myrank , res );
            return -1;
            }
        ppm_map_part_push_1di( vid , vid_len , ppm_npart , &res , 0 );
        if ( res != 0 ) {
            printf( "main[%i]: call to ppm_map_part_push_1di failed with error %i.\n" , myrank , res );
            return -1;
            }
        ppm_map_part_send( &ppm_npart , &ppm_mpart , &res );
        if ( res != 0 ) {
            printf( "main[%i]: call to ppm_map_part_send failed with error %i.\n" , myrank , res );
            return -1;
            }
        ppm_map_part_pop_1di( &vid , &vid_len , &ppm_npart , &ppm_mpart , &res );
        if ( res != 0 ) {
            printf( "main[%i]: call to ppm_map_part_pop_1di failed with error %i.\n" , myrank , res );
            return -1;
            }
        ppm_map_part_pop_1di( &pid , &pid_len , &ppm_npart , &ppm_mpart , &res );
        if ( res != 0 ) {
            printf( "main[%i]: call to ppm_map_part_pop_1di failed with error %i.\n" , myrank , res );
            return -1;
            }
        ppm_map_part_pop_2dd( &vp , 3 , &vp_len , &ppm_npart , &ppm_mpart , &res );
        if ( res != 0 ) {
            printf( "main[%i]: call to ppm_map_part_pop_2dd failed with error %i.\n" , myrank , res );
            return -1;
            }
        ppm_map_part_pop_2dd( &xp , 3 , &xp_len , &ppm_npart , &ppm_mpart , &res );
        if ( res != 0 ) {
            printf( "main[%i]: call to ppm_map_part_pop_2dd failed with error %i.\n" , myrank , res );
            return -1;
            }
        ppm_npart = ppm_mpart;
        timers[tid_pexch0] = getticks() - tic;
        if ( verbose && myrank == 0 ) {
            printf("main[%i]: particle exchange took %.3f ms.\n",myrank,(double)timers[tid_pexch0] * itpms); fflush(stdout);
            }
        if ( 0 && verbose ) {
            printf( "main[%i]: now have %i parts and %i ghosts.\n" , myrank , ppm_npart , ppm_mpart-ppm_npart );
            }
    

        /* Load the engine with the mapped strays. */
        tic = getticks();
        free( ptype );
        if ( ( ptype = (int *)malloc( sizeof(double) * ppm_npart ) ) == NULL ) {
            printf("main[%i]: failed to re-allocate ptype.\n",myrank);
            return -1;
            }
        for ( k = 0 ; k < ppm_npart ; k++ )
            ptype[k] = ( pid[k] % 3 != 0 );
        if ( ( res = engine_load( &e , xp , vp , ptype , pid , vid , NULL , NULL , ppm_npart ) ) < 0 ) {
            printf("main[%i]: engine_load failed with engine_err=%i.\n",myrank,engine_err);
            errs_dump(stdout);
            return -1;
            }
        timers[tid_load] += getticks() - tic;
        if ( verbose && myrank == 0 ) {
            printf("main[%i]: loading strays (%i) took %.3f ms.\n",myrank,ppm_npart,(double)timers[tid_load] * itpms); fflush(stdout);
            }
            

        /* Pack potential ghost particles (SHAKE) into xp, vp and pid. */
        tic = getticks();
        if ( engine_flush_ghosts( &e ) < 0 ) {
            printf("main[%i]: engine_flush_ghosts failed with engine_err=%i.\n",myrank,engine_err);
            errs_dump(stdout);
            return -1;
            }
        if ( ( ppm_npart = engine_unload_marked( &e , xp , vp , NULL , pid , vid , NULL , NULL , NULL , xp_len ) ) < 0 ) {
            printf("main[%i]: engine_unload_marked failed with engine_err=%i.\n",myrank,engine_err);
            errs_dump(stdout);
            return -1;
            }
        ppm_mpart = ppm_npart;
        timers[tid_uload] += getticks() - tic;
        if ( verbose && myrank == 0 ) {
            printf("main[%i]: unloading potential ghosts (SHAKE, %i) took %.3f ms.\n",myrank,ppm_npart,(double)timers[tid_uload] * itpms); fflush(stdout);
            }
            
    
        /* Exchange position data for SHAKE. */
        timers[tid_gexch1] = getticks();
        ppm_map_part_ghost_get( topoid , xp , 3 , xp_len , ppm_npart , 0 , 0.2 , &res );
        if ( res != 0 ) {
            printf( "main[%i]: call to ppm_map_part_ghost_get failed with error %i.\n" , myrank , res );
            return -1;
            }
        ppm_map_part_push_2dd( vp , 3 , vp_len , ppm_npart , &res , 0 );
        if ( res != 0 ) {
            printf( "main[%i]: call to ppm_map_part_push_2dd failed with error %i.\n" , myrank , res );
            return -1;
            }
        ppm_map_part_push_1di( pid , pid_len , ppm_npart , &res , 0 );
        if ( res != 0 ) {
            printf( "main[%i]: call to ppm_map_part_push_1di failed with error %i.\n" , myrank , res );
            return -1;
            }
        ppm_map_part_push_1di( vid , vid_len , ppm_npart , &res , 0 );
        if ( res != 0 ) {
            printf( "main[%i]: call to ppm_map_part_push_1di failed with error %i.\n" , myrank , res );
            return -1;
            }
        ppm_map_part_send( &ppm_npart , &ppm_mpart , &res );
        if ( res != 0 ) {
            printf( "main[%i]: call to ppm_map_part_send failed with error %i.\n" , myrank , res );
            return -1;
            }
        ppm_map_part_pop_1di( &vid , &vid_len , &ppm_npart , &ppm_mpart , &res );
        if ( res != 0 ) {
            printf( "main[%i]: call to ppm_map_part_pop_1di failed with error %i.\n" , myrank , res );
            return -1;
            }
        ppm_map_part_pop_1di( &pid , &pid_len , &ppm_npart , &ppm_mpart , &res );
        if ( res != 0 ) {
            printf( "main[%i]: call to ppm_map_part_pop_1di failed with error %i.\n" , myrank , res );
            return -1;
            }
        ppm_map_part_pop_2dd( &vp , 3 , &vp_len , &ppm_npart , &ppm_mpart , &res );
        if ( res != 0 ) {
            printf( "main[%i]: call to ppm_map_part_pop_2dd failed with error %i.\n" , myrank , res );
            return -1;
            }
        ppm_map_part_pop_2dd( &xp , 3 , &xp_len , &ppm_npart , &ppm_mpart , &res );
        if ( res != 0 ) {
            printf( "main[%i]: call to ppm_map_part_pop_2dd failed with error %i.\n" , myrank , res );
            return -1;
            }
        timers[tid_gexch1] = getticks() - timers[tid_gexch1];
        if ( verbose && myrank == 0 ) {
            printf("main[%i]: ghost exchange (SHAKE) took %.3f ms.\n",myrank,(double)timers[tid_gexch1] * itpms); fflush(stdout);
            }
        /* printf( "main[%i]: now have %i parts and %i ghosts.\n" , myrank , ppm_npart , ppm_mpart-ppm_npart ); */
        
        
        /* Load the ghosts onto the engine. */
        tic = getticks();
        free( ptype );
        if ( ( ptype = (int *)malloc( sizeof(double) * ppm_mpart ) ) == NULL ) {
            printf("main[%i]: failed to re-allocate ptype.\n",myrank);
            return -1;
            }
        for ( k = ppm_npart ; k < ppm_mpart ; k++ )
            ptype[k] = ( pid[k] % 3 != 0 );
        if ( ( res = engine_load_ghosts( &e , &xp[3*ppm_npart] , &vp[3*ppm_npart] , &ptype[ppm_npart] , &pid[ppm_npart] , &vid[ppm_npart] , NULL , NULL , ppm_mpart-ppm_npart ) ) < 0 ) {
            printf("main[%i]: engine_load failed with engine_err=%i.\n",myrank,engine_err);
            errs_dump(stdout);
            return -1;
            }
        timers[tid_load] += getticks() - tic;
        if ( verbose && myrank == 0 ) {
            printf("main[%i]: loading ghosts (SHAKE, %i) took %.3f ms.\n",myrank,ppm_mpart-ppm_npart,(double)timers[tid_load] * itpms); fflush(stdout);
            }
            
        /* Dump the particles. */
        /* sprintf(fname,"parts_%03i.dump",myrank);
        dump = fopen(fname,"w");
        for ( k = ppm_npart ; k < ppm_mpart ; k++ )
            fprintf(dump,"%e %e %e\n",xp[3*k+0],xp[3*k+1],xp[3*k+2]);
        fclose(dump); */
        
        
        /* Resolve particle global/local IDs. */
        tic = getticks();
        bzero( globloc , sizeof(void *) * 2 * nr_parts );
        #pragma omp parallel for schedule(static,100), private(cid,k,p)
        for ( cid = 0 ; cid < e.s.nr_cells ; cid++ )
            if ( !(e.s.cells[cid].flags & cell_flag_ghost) )
                for ( k = 0 ; k < e.s.cells[cid].count ; k++ ) {
                    p = &e.s.cells[cid].parts[k];
                    globloc[p->id].p = p;
                    globloc[p->id].c = &e.s.cells[cid];
                    }
        #pragma omp parallel for schedule(static,100), private(cid,k,p)
        for ( cid = 0 ; cid < e.s.nr_cells ; cid++ )
            if ( e.s.cells[cid].flags & cell_flag_ghost )
                for ( k = 0 ; k < e.s.cells[cid].count ; k++ ) {
                    p = &e.s.cells[cid].parts[k];
                    if ( globloc[p->id].p == NULL ) {
                        globloc[p->id].p = p;
                        globloc[p->id].c = &e.s.cells[cid];
                        }
                    }
        timers[tid_resolv] = getticks() - tic;
        if ( verbose && myrank == 0 ) {
            printf("main[%i]: resolving global/local IDs took %.3f ms.\n",myrank,(double)timers[tid_resolv] * itpms); fflush(stdout);
            }
            
        
        /* Shake the particle positions. */
        tic = getticks();
        #pragma omp parallel for schedule(dynamic,100), private(p_O,p_H1,p_H2,c_O,c_H1,c_H2,vp_O,vp_H1,vp_H2,k,new_O,new_H1,new_H2,old_O,old_H1,old_H2,v_OH1,v_OH2,v_HH,d_OH1,lambda,d_OH2,d_HH)
        for ( j = 0 ; j < nr_mols ; j++ ) {
        
            /* Do we even have the jth molecule? */
            p_O = globloc[j*3].p;
            p_H1 = globloc[j*3+1].p;
            p_H2 = globloc[j*3+2].p;
            if ( p_O == NULL || p_H1 == NULL || p_H2 == NULL ||
                ( p_O->flags & part_flag_ghost && p_H1->flags & part_flag_ghost && p_H2->flags & part_flag_ghost ) )
                continue;
        
            // unwrap the data
            c_O = globloc[j*3].c; c_H1 = globloc[j*3+1].c; c_H2 = globloc[j*3+2].c;
            for ( k = 0 ; k < 3 ; k++ ) {
                new_O[k] = p_O->x[k] + c_O->origin[k];
                vp_O[k] = p_O->v[k];
                new_H1[k] = p_H1->x[k] + c_H1->origin[k];
                vp_H1[k] = p_H1->v[k];
                new_H2[k] = p_H2->x[k] + c_H2->origin[k];
                vp_H2[k] = p_H2->v[k];
                }
            for ( k = 0 ; k < 3 ; k++ ) {
                old_O[k] = new_O[k] - e.dt * vp_O[k];
                if ( new_H1[k] - new_O[k] > ldim[k] * 0.5 )
                    new_H1[k] -= ldim[k];
                else if ( new_H1[k] - new_O[k] < -ldim[k] * 0.5 )
                    new_H1[k] += ldim[k];
                old_H1[k] = new_H1[k] - e.dt * vp_H1[k];
                if ( new_H2[k] - new_O[k] > ldim[k] * 0.5 )
                    new_H2[k] -= ldim[k];
                else if ( new_H2[k] - new_O[k] < -ldim[k] * 0.5 )
                    new_H2[k] += ldim[k];
                old_H2[k] = new_H2[k] - e.dt * vp_H2[k];
                v_OH1[k] = old_O[k] - old_H1[k];
                v_OH2[k] = old_O[k] - old_H2[k];
                v_HH[k] = old_H1[k] - old_H2[k];
                }
                
            // main loop
            while ( 1 ) {
            
                // correct for the OH1 constraint
                for ( d_OH1 = 0.0 , k = 0 ; k < 3 ; k++ )
                    d_OH1 += (new_O[k] - new_H1[k]) * (new_O[k] - new_H1[k]);
                lambda = 0.5 * ( 0.1*0.1 - d_OH1 ) /
                    ( (new_O[0] - new_H1[0]) * v_OH1[0] + (new_O[1] - new_H1[1]) * v_OH1[1] + (new_O[2] - new_H1[2]) * v_OH1[2] );
                for ( k = 0 ; k < 3 ; k++ ) {
                    new_O[k] += lambda * 1.00794 / ( 1.00794 + 15.9994 ) * v_OH1[k];
                    new_H1[k] -= lambda * 15.9994 / ( 1.00794 + 15.9994 ) * v_OH1[k];
                    }
                    
                // correct for the OH2 constraint
                for ( d_OH2 = 0.0 , k = 0 ; k < 3 ; k++ )
                    d_OH2 += (new_O[k] - new_H2[k]) * (new_O[k] - new_H2[k]);
                lambda = 0.5 * ( 0.1*0.1 - d_OH2 ) /
                    ( (new_O[0] - new_H2[0]) * v_OH2[0] + (new_O[1] - new_H2[1]) * v_OH2[1] + (new_O[2] - new_H2[2]) * v_OH2[2] );
                for ( k = 0 ; k < 3 ; k++ ) {
                    new_O[k] += lambda * 1.00794 / ( 1.00794 + 15.9994 ) * v_OH2[k];
                    new_H2[k] -= lambda * 15.9994 / ( 1.00794 + 15.9994 ) * v_OH2[k];
                    }
                    
                // correct for the HH constraint
                for ( d_HH = 0.0 , k = 0 ; k < 3 ; k++ )
                    d_HH += (new_H1[k] - new_H2[k]) * (new_H1[k] - new_H2[k]);
                lambda = 0.5 * ( 0.1633*0.1633 - d_HH ) /
                    ( (new_H1[0] - new_H2[0]) * v_HH[0] + (new_H1[1] - new_H2[1]) * v_HH[1] + (new_H1[2] - new_H2[2]) * v_HH[2] );
                for ( k = 0 ; k < 3 ; k++ ) {
                    new_H1[k] += lambda * 0.5 * v_HH[k];
                    new_H2[k] -= lambda * 0.5 * v_HH[k];
                    }
                    
                // check the tolerances
                if ( fabs( d_OH1 - 0.1*0.1 ) < 1e-6 &&
                    fabs( d_OH2 - 0.1*0.1 ) < 1e-6 &&  
                    fabs( d_HH - 0.1633*0.1633 ) < 1e-6 )
                    break;
                    
                // printf("main: mol %i: d_OH1=%e, d_OH2=%e, d_HH=%e.\n",j,sqrt(d_OH1),sqrt(d_OH2),sqrt(d_HH));
                // getchar();
                    
                }
                
            // wrap the positions back
            for ( k = 0 ; k < 3 ; k++ ) {
            
                // write O
                p_O->x[k] = new_O[k] - c_O->origin[k];
                p_O->v[k] = (new_O[k] - old_O[k]) / e.dt;
                
                // write H1
                p_H1->x[k] -= e.dt * p_H1->v[k];
                p_H1->v[k] = (new_H1[k] - old_H1[k]) / e.dt;
                p_H1->x[k] += e.dt * p_H1->v[k];
                
                // write H2
                p_H2->x[k] -= e.dt * p_H2->v[k];
                p_H2->v[k] = (new_H2[k] - old_H2[k]) / e.dt;
                p_H2->x[k] += e.dt * p_H2->v[k];
                
                }
                
            } // shake molecules
        timers[tid_shake] = getticks() - tic;
        if ( verbose && myrank == 0 ) {
            printf("main[%i]: SHAKE took %.3f ms.\n",myrank,(double)timers[tid_shake] * itpms); fflush(stdout);
            }
        
        
        /* Dump the particles. */
        /* sprintf(fname,"parts_%03i.dump",myrank);
        dump = fopen(fname,"w");
        for ( k = 0 ; k < ppm_mpart ; k++ )
            fprintf(dump,"%e %e %e\n",xp[3*k+0],xp[3*k+1],xp[3*k+2]);
        fclose(dump); */
        
        
        /* Resolve particle global/local IDs. */
        /* tic = getticks();
        for ( k = 0 ; k < nr_parts ; k++ )
            globloc[k] = -1;
        for ( k = 0 ; k < ppm_npart ; k++ )
            globloc[ pid[k] ] = k;
        for ( k = ppm_npart ; k < ppm_mpart ; k++ )
            if ( globloc[ pid[k] ] < 0 )
                globloc[ pid[k] ] = k;
        toc = getticks();
        if ( verbose && myrank == 0 ) {
            printf("main[%i]: resolving global/local IDs took %.3f ms.\n",myrank,(double)(toc-tic) * itpms); fflush(stdout);
            } */
            
        
        /* Compute the system temperature. */
        tic = getticks();
        
        /* Get the total atomic kinetic energy, v_com and molecular kinetic energy. */
        ekin = 0.0; epot = 0.0;
        vcom_tot_x = 0.0; vcom_tot_y = 0.0; vcom_tot_z = 0.0;
        temp = 0.0;
        #pragma omp parallel for schedule(static), private(p,p_O,p_H1,p_H2,j,k,vcom,v2), reduction(+:ekin,epot,vcom_tot_x,vcom_tot_y,vcom_tot_z,temp)
        for ( cid = 0 ; cid < e.s.nr_cells ; cid++ ) {
            epot += e.s.cells[cid].epot;
            if ( !(e.s.cells[cid].flags & cell_flag_ghost ) )
                for ( j = 0 ; j < e.s.cells[cid].count ; j++ ) {
                    p = &( e.s.cells[cid].parts[j] );
                    v2 = p->v[0]*p->v[0] + p->v[1]*p->v[1] + p->v[2]*p->v[2];
                    if ( p->type == 0 )
                        ekin += v2 * 15.9994 * 0.5;
                    else
                        ekin += v2 * 1.00794 * 0.5;
                    if ( p->type != 0 )
                        continue;
                    p_O = p; p_H1 = globloc[ p_O->id + 1 ].p; p_H2 = globloc[ p_O->id + 2 ].p;
                    for ( k = 0 ; k < 3 ; k++ )
                        vcom[k] = ( p_O->v[k] * 15.9994 +
                            p_H1->v[k] * 1.00794 +
                            p_H2->v[k] * 1.00794 ) / 1.801528e+1;
                    vcom_tot_x += vcom[0]; vcom_tot_y += vcom[1]; vcom_tot_z += vcom[2];
                    temp += 9.00764 * ( vcom[0]*vcom[0] + vcom[1]*vcom[1] + vcom[2]*vcom[2] );
                    }
            }
        vcom_tot[0] = vcom_tot_x; vcom_tot[1] = vcom_tot_y; vcom_tot[2] = vcom_tot_z;
        vcom_tot[3] = temp;
            
        /* Collect vcom and ekin from all procs and compute the temp. */
        vcom_tot[4] = epot; vcom_tot[5] = ekin;
        if ( ( res = MPI_Allreduce( MPI_IN_PLACE , vcom_tot , 6 , MPI_DOUBLE_PRECISION , MPI_SUM , MPI_COMM_WORLD ) ) != MPI_SUCCESS ) {
            printf( "main[%i]: call to MPI_Allreduce failed with error %i.\n" , myrank , res );
            return -1;
            }
        ekin = vcom_tot[5]; epot = vcom_tot[4];
        for ( k = 0 ; k < 3 ; k++ )
            vcom_tot[k] /= nr_mols * 1.801528e+1;
        temp = vcom_tot[3] / ( 1.5 * 6.022045E23 * 1.380662E-26 * nr_mols );
        w = sqrt( 1.0 + 0.1 * ( 300.0 / temp - 1.0 ) );
        // printf("main[%i]: vcom_tot is [ %e , %e , %e ].\n",myrank,vcom_tot[0],vcom_tot[1],vcom_tot[2]); fflush(stdout);
            
        /* Subtract the vcom from the molecules on this proc. */
        #pragma omp parallel for schedule(static), private(j,p_O,p_H1,p_H2,k,vcom)
        for ( cid = 0 ; cid < e.s.nr_cells ; cid++ )
            for ( j = 0 ; j < e.s.cells[cid].count ; j++ ) {
                p_O = &( e.s.cells[cid].parts[j] );
                if ( ( p_O->type != 0 ) ||
                     ( p_O != globloc[ p_O->id ].p ) ||
                     ( p_H1 = globloc[ p_O->id + 1 ].p ) == NULL ||
                     ( p_H2 = globloc[ p_O->id + 2 ].p ) == NULL )
                    continue;
                for ( k = 0 ; k < 3 ; k++ ) {
                    vcom[k] = ( p_O->v[k] * 15.9994 +
                        p_H1->v[k] * 1.00794 +
                        p_H2->v[k] * 1.00794 ) / 1.801528e+1;
                    vcom[k] -= vcom_tot[k];
                    vcom[k] *= ( w - 1.0 );
                    p_O->v[k] += vcom[k];
                    p_H1->v[k] += vcom[k];
                    p_H2->v[k] += vcom[k];
                    }
                }
        timers[tid_temp] = getticks() - tic;
        if ( verbose && myrank == 0 ) {
            printf("main[%i]: thermostat took %.3f ms.\n",myrank,(double)timers[tid_temp] * itpms); fflush(stdout);
            }
                        
        
        /* Unload any stray particles. */
        tic = getticks();
        if ( space_shuffle( &e.s ) < 0 ) {
            printf("main[%i]: space_shuffle failed with space_err=%i.\n",myrank,space_err);
            errs_dump(stdout);
            return -1;
            }
        if ( ( ppm_npart = engine_unload_strays( &e , xp , vp , NULL , pid , vid , NULL , NULL , NULL , xp_len ) ) < 0 ) {
            printf("main[%i]: engine_unload_strays failed with engine_err=%i.\n",myrank,engine_err);
            errs_dump(stdout);
            return -1;
            }
        ppm_mpart = ppm_npart;
        timers[tid_uload] += getticks() - tic;
        if ( verbose && myrank == 0 ) {
            printf("main[%i]: unloading strays (%i) took %.3f ms.\n",myrank,ppm_npart,(double)timers[tid_uload] * itpms); fflush(stdout);
            }
        
        /* Re-distribute the particles to the processors. */
        tic = getticks();
        /* for ( j = 0 ; j < ppm_npart ; j++ )
            for ( k = 0 ; k < 3 ; k++ )
                if ( xp[j*3+k] < origin[k] )
                    xp[j*3+k] += dim[k];
                else if ( xp[j*3+k] > dim[k] )
                    xp[j*3+k] -= dim[k]; */
        ppm_impose_part_bc( topoid , xp , xp_len , ppm_npart , &res );
        if ( res != 0 ) {
            printf( "main[%i]: call to ppm_impose_part_bc failed with error %i.\n" , myrank , res );
            return -1;
            }
        ppm_map_part_partial( topoid , xp , xp_len , ppm_npart , &res );
        if ( res != 0 ) {
            printf( "main[%i]: call to ppm_map_part_partial failed with error %i.\n" , myrank , res );
            return -1;
            }
        ppm_map_part_push_2dd( vp , 3 , vp_len , ppm_npart , &res , 0 );
        if ( res != 0 ) {
            printf( "main[%i]: call to ppm_map_part_push_2dd failed with error %i.\n" , myrank , res );
            return -1;
            }
        ppm_map_part_push_1di( pid , pid_len , ppm_npart , &res , 0 );
        if ( res != 0 ) {
            printf( "main[%i]: call to ppm_map_part_push_1di failed with error %i.\n" , myrank , res );
            return -1;
            }
        ppm_map_part_push_1di( vid , vid_len , ppm_npart , &res , 0 );
        if ( res != 0 ) {
            printf( "main[%i]: call to ppm_map_part_push_1di failed with error %i.\n" , myrank , res );
            return -1;
            }
        ppm_map_part_send( &ppm_npart , &ppm_mpart , &res );
        if ( res != 0 ) {
            printf( "main[%i]: call to ppm_map_part_send failed with error %i.\n" , myrank , res );
            return -1;
            }
        ppm_map_part_pop_1di( &vid , &vid_len , &ppm_npart , &ppm_mpart , &res );
        if ( res != 0 ) {
            printf( "main[%i]: call to ppm_map_part_pop_1di failed with error %i.\n" , myrank , res );
            return -1;
            }
        ppm_map_part_pop_1di( &pid , &pid_len , &ppm_npart , &ppm_mpart , &res );
        if ( res != 0 ) {
            printf( "main[%i]: call to ppm_map_part_pop_1di failed with error %i.\n" , myrank , res );
            return -1;
            }
        ppm_map_part_pop_2dd( &vp , 3 , &vp_len , &ppm_npart , &ppm_mpart , &res );
        if ( res != 0 ) {
            printf( "main[%i]: call to ppm_map_part_pop_2dd failed with error %i.\n" , myrank , res );
            return -1;
            }
        ppm_map_part_pop_2dd( &xp , 3 , &xp_len , &ppm_npart , &ppm_mpart , &res );
        if ( res != 0 ) {
            printf( "main[%i]: call to ppm_map_part_pop_2dd failed with error %i.\n" , myrank , res );
            return -1;
            }
        ppm_npart = ppm_mpart;
        timers[tid_pexch1] = getticks() - tic;
        if ( verbose && myrank == 0 ) {
            printf("main[%i]: particle exchange took %.3f ms.\n",myrank,(double)timers[tid_pexch1] * itpms); fflush(stdout);
            }
        if ( 0 && verbose ) {
            printf( "main[%i]: now have %i parts and %i ghosts.\n" , myrank , ppm_npart , ppm_mpart-ppm_npart );
            fflush(stdout);
            }
    

        /* Load the engine with the mapped strays. */
        tic = getticks();
        free( ptype );
        if ( ( ptype = (int *)malloc( sizeof(double) * ppm_npart ) ) == NULL ) {
            printf("main[%i]: failed to re-allocate ptype.\n",myrank);
            return -1;
            }
        for ( k = 0 ; k < ppm_npart ; k++ )
            ptype[k] = ( pid[k] % 3 != 0 );
        if ( ( res = engine_load( &e , xp , vp , ptype , pid , vid , NULL , NULL , ppm_npart ) ) < 0 ) {
            printf("main[%i]: engine_load failed with engine_err=%i.\n",myrank,engine_err);
            errs_dump(stdout);
            return -1;
            }
        timers[tid_load] += getticks() - tic;
        if ( verbose && myrank == 0 ) {
            printf("main[%i]: loading strays (%i) took %.3f ms.\n",myrank,ppm_npart,(double)timers[tid_load] * itpms); fflush(stdout);
            }
            
            
        /* Drop a line. */
        toc_step = getticks();
        if ( myrank == 0 ) {
            /* printf("%i %e %e %e %i %i %.3f ms\n",
                e.time,epot,ekin,temp,e.s.nr_swaps,e.s.nr_stalls,(double)(toc_step-tic_step) * itpms); fflush(stdout); */
            printf("%i %e %e %e %i %i %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f ms\n",
                e.time,epot,ekin,temp,e.s.nr_swaps,e.s.nr_stalls,(toc_step-tic_step) * itpms,
                timers[0]*itpms, timers[1]*itpms, timers[2]*itpms, timers[3]*itpms, timers[4]*itpms, 
                timers[5]*itpms, timers[6]*itpms, timers[7]*itpms, timers[8]*itpms, timers[9]*itpms ); fflush(stdout);
            }
        
        
        } /* main loop. */
        
    
    /* Exit gracefuly. */
    if ( ( res = MPI_Finalize() ) != MPI_SUCCESS ) {
        printf( "main[%i]: call to MPI_Finalize failed with error %i.\n" , myrank , res );
        return -1;
        }
    fflush(stdout);
    printf( "main[%i]: exiting.\n" , myrank );
    return 0;

    }
