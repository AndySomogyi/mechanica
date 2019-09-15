/*******************************************************************************
 * This file is part of mdcore.
 * Coypright (c) 2010 Pedro Gonnet (gonnet@maths.ox.ac.uk)
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
#include "../config.h"

/* MPI headers. */
#ifdef WITH_MPI
    #include <mpi.h>
#endif

/* OpenMP headers. */
#include <omp.h>

/* FFTW3 headers. */
#ifdef HAVE_FFTW3
    #include <complex.h>
    #include <fftw3.h>
#endif

/* Include mdcore. */
#include "mdcore.h"

/* Ticks Per Second. */
#ifndef CPU_TPS
    #define CPU_TPS 2.67e+9
#endif

/* Engine flags? */
#ifndef ENGINE_FLAGS
    #define ENGINE_FLAGS engine_flag_parbonded
#endif


int main ( int argc , char *argv[] ) {

    const double origin[3] = { 0.0 , 0.0 , 0.0 };
    // double dim[3] = { 3.166 , 3.166 , 3.166 };
    // int nr_mols = 1000;
    // double dim[3] = { 6.332 , 6.332 , 6.332 };
    // int nr_mols = 8000;
    // const double dim[3] = { 4.0 , 4.0 , 4.0 };
    // const int nr_mols = 2016;
    double dim[3] = { 8.0 , 8.0 , 8.0 };
    int nr_mols = 16128;
    // double dim[3] = { 16.0 , 16.0 , 16.0 };
    // int nr_mols = 129024;
    int nr_parts = nr_mols * 3;
    double Temp = 300.0;
    
    double x[3], v2;
    double epot, ekin, temp, cutoff = 1.0, cellwidth;
    FPTYPE ee, eff;
    struct engine e;
    struct particle *p, pO, pH;
    struct potential *pot_OO, *pot_OH, *pot_HH, *pot_OHb, *pot_HOH;
    // struct potential *pot_ee;
    int i, j, k, cid, pid, nr_runners = 1, nr_steps = 1000;
    int nx, ny, nz;
    double hx, hy, hz;
    double vtot[3] = { 0.0 , 0.0 , 0.0 }, w, mass_tot;
    // struct cellpair cp;
    FILE *psf, *pdb;
    char fname[100];
    ticks tic, toc, toc_step, toc_bonded, toc_temp;
    double L[] = { cutoff , cutoff , cutoff };
    
    
    /* Start the clock... */
    tic = getticks();
    
    /* Trap on all floating-point problems. */
    // feenableexcept( FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW );
    
    // did the user supply a cutoff?
    if ( argc > 4 ) {
        cellwidth = atof( argv[4] );
        nr_mols *= ( cellwidth * cellwidth * cellwidth );
        for ( k = 0 ; k < 3 ; k++ ) {
            L[k] = cellwidth;
            dim[k] *= cellwidth * (1.0 + DBL_EPSILON);
            }
        }
    else
        cellwidth = 1.0;
    printf("main: cell width set to %22.16e.\n", cellwidth);
    
    // initialize the engine
    printf("main: initializing the engine... "); fflush(stdout);
    if ( engine_init( &e , origin , dim , L , cutoff , space_periodic_full , 3 , ENGINE_FLAGS | engine_flag_verlet_pairwise ) != 0 ) {
        printf("main: engine_init failed with engine_err=%i.\n",engine_err);
        errs_dump(stdout);
        return 1;
        }
    printf("done.\n"); fflush(stdout);
    
    // set the interaction cutoff
    printf("main: cell dimensions = [ %i , %i , %i ].\n", e.s.cdim[0] , e.s.cdim[1] , e.s.cdim[2] );
    printf("main: cell size = [ %e , %e , %e ].\n" , e.s.h[0] , e.s.h[1] , e.s.h[2] );
    printf("main: cutoff set to %22.16e.\n", cutoff);
    printf("main: nr tasks: %i.\n",e.s.nr_tasks);
        
    /* mix-up the pair list just for kicks
    printf("main: shuffling the interaction pairs... "); fflush(stdout);
    srand(6178);
    for ( i = 0 ; i < e.s.nr_pairs ; i++ ) {
        j = rand() % e.s.nr_pairs;
        if ( i != j ) {
            cp = e.s.pairs[i];
            e.s.pairs[i] = e.s.pairs[j];
            e.s.pairs[j] = cp;
            }
        }
    printf("done.\n"); fflush(stdout); */
        

    // initialize the O-H potential
    if ( ( pot_OH = potential_create_Ewald( 0.1 , 1.0 , -0.35921288 , 3.0 , 1.0e-4 ) ) == NULL ) {
        printf("main: potential_create_Ewald failed with potential_err=%i.\n",potential_err);
        errs_dump(stdout);
        return 1;
        }
    printf("main: constructed OH-potential with %i intervals.\n",pot_OH->n); fflush(stdout);

    // initialize the H-H potential
    if ( ( pot_HH = potential_create_Ewald( 0.1 , 1.0 , 1.7960644e-1 , 3.0 , 1.0e-4 ) ) == NULL ) {
        printf("main: potential_create_Ewald failed with potential_err=%i.\n",potential_err);
        errs_dump(stdout);
        return 1;
        }
    printf("main: constructed HH-potential with %i intervals.\n",pot_HH->n); fflush(stdout);

    // initialize the O-O potential
    if ( ( pot_OO = potential_create_LJ126_Ewald( 0.25 , 1.0 , 2.637775819766153e-06 , 2.619222661792581e-03 , 7.1842576e-01 , 3.0 , 0.9e-4 ) ) == NULL ) {
        printf("main: potential_create_LJ126_Ewald failed with potential_err=%i.\n",potential_err);
        errs_dump(stdout);
        return 1;
        }
    printf("main: constructed OO-potential with %i intervals.\n",pot_OO->n); fflush(stdout);
    
    // initialize the O-H bond potential
    if ( ( pot_OHb = potential_create_harmonic( 0.05 , 0.15 , 463700/2 , 0.1 , 1.0e-3 ) ) == NULL ) {
        printf("main: potential_create_harmonic_bond failed with potential_err=%i.\n",potential_err);
        errs_dump(stdout);
        return 1;
        }
    printf("main: constructed OH bonded potential with %i intervals.\n",pot_OHb->n); fflush(stdout);
    
    // initialize the H-O-H angle potential
    if ( ( pot_HOH = potential_create_harmonic_angle( (109.47-60)/180*M_PI , (109.47+60)/180*M_PI , 383.0/2 , 109.47/180*M_PI , 1.0e-4 ) ) == NULL ) {
        printf("main: potential_create_harmonic_angle failed with potential_err=%i.\n",potential_err);
        errs_dump(stdout);
        return 1;
        }
    printf("main: constructed HOH angle potential with %i intervals.\n",pot_HOH->n); fflush(stdout);
        
    /* dump the OO-potential to make sure its ok... */
    /* for ( i = 0 ; i < 1000 ; i++ ) {
        temp = 0.1 + (double)i/1000 * 0.9;
        potential_eval( pot_HH , temp*temp , &ee , &eff );
        printf("%23.16e %23.16e %23.16e %23.16e\n", temp , ee , eff , 1.7960644e-1 * potential_Ewald_p( temp , 3.0 ) );
        }
    return 0; */
        
    
    /* register the particle types. */
    if ( engine_addtype( &e , 15.9994 , -0.8476 , "OT" , "OH2" ) < 0 ||
         engine_addtype( &e , 1.00794 , 0.4238 , "HT" , "H1" ) < 0 ||
         engine_addtype( &e , 1.00794 , 0.4238 , "HT" , "H2" ) < 0 ) {
        printf("main: call to engine_addtype failed.\n");
        errs_dump(stdout);
        return 1;
        }
        
    // register these potentials.
    if ( engine_addpot( &e , pot_OO , 0 , 0 ) < 0 ||
         engine_addpot( &e , pot_HH , 1 , 1 ) < 0 ||
         engine_addpot( &e , pot_HH , 1 , 2 ) < 0 ||
         engine_addpot( &e , pot_HH , 2 , 2 ) < 0 ||
         engine_addpot( &e , pot_OH , 0 , 1 ) < 0 ||
         engine_addpot( &e , pot_OH , 0 , 2 ) < 0 ) {
        printf("main: call to engine_addpot failed.\n");
        errs_dump(stdout);
        return 1;
        }
    if ( engine_bond_addpot( &e , pot_OHb , 0 , 1 ) < 0 ) {
        printf("main: call to engine_addbondpot failed.\n");
        errs_dump(stdout);
        return 1;
        }
    if ( engine_angle_addpot( &e , pot_HOH ) < 0 ) {
        printf("main: call to engine_addanglepot failed.\n");
        errs_dump(stdout);
        return 1;
        }
    
    // set fields for all particles
    srand(6178);
    pO.type = 0;
    pH.type = 1;
    pO.flags = PARTICLE_FLAG_NONE;
    pH.flags = PARTICLE_FLAG_NONE;
    for ( k = 0 ; k < 3 ; k++ ) {
        pO.v[k] = 0.0; pH.v[k] = 0.0;
        pO.f[k] = 0.0; pH.f[k] = 0.0;
        }
    #ifdef VECTORIZE
        pO.v[3] = 0.0; pO.f[3] = 0.0; pO.x[3] = 0.0;
        pH.v[3] = 0.0; pH.f[3] = 0.0; pH.x[3] = 0.0;
    #endif
    
    // create and add the particles
    printf("main: initializing particles... "); fflush(stdout);
    nx = ceil( pow( nr_mols , 1.0/3 ) ); hx = dim[0] / nx;
    ny = ceil( sqrt( ((double)nr_mols) / nx ) ); hy = dim[1] / ny;
    nz = ceil( ((double)nr_mols) / nx / ny ); hz = dim[2] / nz;
    for ( i = 0 ; i < nx ; i++ ) {
        x[0] = 0.07 + i * hx;
        for ( j = 0 ; j < ny ; j++ ) {
            x[1] = 0.07 + j * hy;
            for ( k = 0 ; k < nz && k + nz * ( j + ny * i ) < nr_mols ; k++ ) {
                pO.vid = (k + nz * ( j + ny * i ));
                pO.id = 3*pO.vid;
                x[2] = 0.07 + k * hz;
                pO.v[0] = ((double)rand()) / RAND_MAX - 0.5;
                pO.v[1] = ((double)rand()) / RAND_MAX - 0.5;
                pO.v[2] = ((double)rand()) / RAND_MAX - 0.5;
                temp = 1.1 / sqrt( pO.v[0]*pO.v[0] + pO.v[1]*pO.v[1] + pO.v[2]*pO.v[2] );
                pO.v[0] *= temp; pO.v[1] *= temp; pO.v[2] *= temp;
                vtot[0] += pO.v[0]*16; vtot[1] += pO.v[1]*16; vtot[2] += pO.v[2]*16;
                if ( space_addpart( &(e.s) , &pO , x ) != 0 ) {
                    printf("main: space_addpart failed with space_err=%i.\n",space_err);
                    errs_dump(stdout);
                    return 1;
                    }
                x[0] += 0.1;
                pH.v[0] = ((double)rand()) / RAND_MAX - 0.5;
                pH.v[1] = ((double)rand()) / RAND_MAX - 0.5;
                pH.v[2] = ((double)rand()) / RAND_MAX - 0.5;
                temp = 1.1 / sqrt( pH.v[0]*pH.v[0] + pH.v[1]*pH.v[1] + pH.v[2]*pH.v[2] );
                pH.v[0] *= temp; pH.v[1] *= temp; pH.v[2] *= temp;
                vtot[0] += pH.v[0]; vtot[1] += pH.v[1]; vtot[2] += pH.v[2];
                pH.vid = pO.vid; pH.id = pO.id+1; pH.type = 1;
                if ( space_addpart( &(e.s) , &pH , x ) != 0 ) {
                    printf("main: space_addpart failed with space_err=%i.\n",space_err);
                    errs_dump(stdout);
                    return 1;
                    }
                x[0] -= 0.13333;
                x[1] += 0.09428;
                pH.v[0] = ((double)rand()) / RAND_MAX - 0.5;
                pH.v[1] = ((double)rand()) / RAND_MAX - 0.5;
                pH.v[2] = ((double)rand()) / RAND_MAX - 0.5;
                temp = 1.1 / sqrt( pH.v[0]*pH.v[0] + pH.v[1]*pH.v[1] + pH.v[2]*pH.v[2] );
                pH.v[0] *= temp; pH.v[1] *= temp; pH.v[2] *= temp;
                vtot[0] += pH.v[0]; vtot[1] += pH.v[1]; vtot[2] += pH.v[2];
                pH.vid = pO.vid; pH.id = pO.id+2; pH.type = 2;
                if ( space_addpart( &(e.s) , &pH , x ) != 0 ) {
                    printf("main: space_addpart failed with space_err=%i.\n",space_err);
                    errs_dump(stdout);
                    return 1;
                    }
                x[0] += 0.03333;
                x[1] -= 0.09428;
                }
            }
        }
    // e.s.partlist[1]->v[2] += 2.0;
    for ( k = 0 ; k < 3 ; k++ )
        vtot[k] /= nr_mols * 18;
    for ( cid = 0 ; cid < e.s.nr_cells ; cid++ )
        for ( pid = 0 ; pid < e.s.cells[cid].count ; pid++ )
            for ( k = 0 ; k < 3 ; k++ )
                e.s.cells[cid].parts[pid].v[k] -= vtot[k] * e.types[e.s.cells[cid].parts[pid].type].mass;
    printf("done.\n"); fflush(stdout);
    printf("main: inserted %i particles.\n", e.s.nr_parts);
    
    
    /* Add the bonds and angles. */
    for ( i = 0 ; i < nr_mols ; i++ ) {
        if ( engine_bond_add( &e , 3*i , 3*i+1 ) < 0 ||
             engine_bond_add( &e , 3*i , 3*i+2 ) < 0 ) {
            printf("main: space_addbond failed with space_err=%i.\n",space_err);
            errs_dump(stdout);
            return 1;
            }
        /* if ( engine_rigid_add( &e , 3*i , 3*i+1 , 0.1 ) < 0 ||
             engine_rigid_add( &e , 3*i , 3*i+2 , 0.1 ) < 0 ) {
            printf("main: engine_rigid_add failed with space_err=%i.\n",engine_err);
            errs_dump(stdout);
            return 1;
            } */
        if ( engine_angle_add( &e , 3*i+1 , 3*i , 3*i+2 , 0 ) < 0 ) {
            printf("main: engine_addangle failed with space_err=%i.\n",engine_err);
            errs_dump(stdout);
            return 1;
            }
        if ( engine_exclusion_add( &e , 3*i , 3*i+1 ) < 0 ||
             engine_exclusion_add( &e , 3*i , 3*i+2 ) < 0 ||
             engine_exclusion_add( &e , 3*i+1 , 3*i+2 ) < 0 ) {
            printf("main: engine_exclusion_add failed with engine_err=%i.\n",engine_err);
            errs_dump(stdout);
            abort();
            }
        /* if ( engine_rigid_add( &e , 3*i , 3*i+2 , 0.1633 ) < 0 ) {
            printf("main: engine_rigid_add failed with space_err=%i.\n",engine_err);
            errs_dump(stdout);
            return 1;
            } */
        }
    if ( engine_exclusion_shrink( &e ) < 0 ) {
        printf("main: engine_exclusion_shrink failed with engine_err=%i.\n",engine_err);
        errs_dump(stdout);
        return -1;
        }
    printf( "main: have %i angles.\n" , e.nr_angles );
                    

    // set the time and time-step by hand
    e.time = 0;
    if ( argc > 3 )
        e.dt = atof( argv[3] );
    else
        e.dt = 0.00025;
    printf("main: dt set to %f fs.\n", e.dt*1000 );
    
    
    /* Shake the particle positions. */
    if ( engine_rigid_eval( &e ) != 0 ) {
        printf("main: engine_rigid_eval failed with engine_err=%i.\n",engine_err);
        errs_dump(stdout);
        return -1;
        }
    mass_tot = 0.0; vtot[0] = 0.0; vtot[1] = 0.0; vtot[2] = 0.0;
    for ( k = 0 ; k < e.s.nr_parts ; k++ ) {
        p = e.s.partlist[k];
        mass_tot += e.types[p->type].mass;
        vtot[0] += p->v[0] * e.types[p->type].mass;
        vtot[1] += p->v[1] * e.types[p->type].mass;
        vtot[2] += p->v[2] * e.types[p->type].mass;
        }
    vtot[0] /= mass_tot; vtot[1] /= mass_tot; vtot[2] /= mass_tot;
    for ( k = 0 ; k < e.s.nr_parts ; k++ ) {
        p = e.s.partlist[k];
        p->v[0] -= vtot[0];
        p->v[1] -= vtot[1];
        p->v[2] -= vtot[2];
        }
        
        
    toc = getticks();
    printf("main: setup took %.3f ms.\n",(double)(toc-tic) * 1000 / CPU_TPS);
    
    // did the user specify a number of runners?
    if ( argc > 1 )
        nr_runners = atoi( argv[1] );
        
    // start the engine
    if ( engine_start( &e , nr_runners , nr_runners ) != 0 ) {
        printf("main: engine_start failed with engine_err=%i.\n",engine_err);
        errs_dump(stdout);
        return 1;
        }
    
    // did the user specify a number of steps?
    if ( argc > 2 )
        nr_steps = atoi( argv[2] );
        
    // do a few steps
    for ( i = 0 ; i < nr_steps ; i++ ) {
    
        // take a step
        tic = getticks();
        if ( engine_step( &e ) != 0 ) {
            printf("main: engine_step failed with engine_err=%i.\n",engine_err);
            errs_dump(stdout);
            return 1;
            }
        toc_step = getticks();
        toc_bonded = getticks();
        

        // get the total COM-velocities and ekin
        epot = e.s.epot; ekin = 0.0;
        vtot[0] = 0.0; vtot[1] = 0.0; vtot[2] = 0.0;
        // #pragma omp parallel for schedule(static,100), private(cid,pid,k,v2), reduction(+:epot,ekin)
        for ( cid = 0 ; cid < e.s.nr_cells ; cid++ ) {
            for ( pid = 0 ; pid < e.s.cells[cid].count ; pid++ ) {
                for ( v2 = 0.0 , k = 0 ; k < 3 ; k++ ) {
                    v2 += e.s.cells[cid].parts[pid].v[k] * e.s.cells[cid].parts[pid].v[k];
                    vtot[k] += e.s.cells[cid].parts[pid].v[k] * e.types[e.s.cells[cid].parts[pid].type].mass;
                    }
                ekin += e.types[e.s.cells[cid].parts[pid].type].mass * v2;
                }
            }
        ekin *= 0.5;
        vtot[0] /= nr_mols*18; vtot[1] /= nr_mols*18; vtot[2] /= nr_mols*18;
        // printf( "main: vtot is [ %e , %e , %e ].\n" , vtot[0] , vtot[1] , vtot[2] );

        // compute the temperature and scaling
        temp = ekin / ( 1.5 * 6.022045E23 * 1.380662E-26 * nr_mols * 3 );
        w = sqrt( 1.0 + 0.1 * ( Temp / temp - 1.0 ) );

        // compute the atomic heat
        if ( i < 10000 ) {
        
            // scale the velocities
            // #pragma omp parallel for schedule(static,100), private(cid,pid,k), reduction(+:epot,ekin)
            for ( cid = 0 ; cid < e.s.nr_cells ; cid++ ) {
                for ( pid = 0 ; pid < e.s.cells[cid].count ; pid++ )
                    for ( k = 0 ; k < 3 ; k++ )
                        e.s.cells[cid].parts[pid].v[k] = w * (e.s.cells[cid].parts[pid].v[k] - vtot[k]);
                }
            
            // re-compute the kinetic energy.
            ekin = 0.0;
            for ( cid = 0 ; cid < e.s.nr_cells ; cid++ ) {
                for ( pid = 0 ; pid < e.s.cells[cid].count ; pid++ ) {
                    for ( v2 = 0.0 , k = 0 ; k < 3 ; k++ )
                        v2 += e.s.cells[cid].parts[pid].v[k] * e.s.cells[cid].parts[pid].v[k];
                    ekin += e.types[ e.s.cells[cid].parts[pid].type ].mass * v2;
                    }
                }
            ekin *= 0.5;
            
            } // apply atomic thermostat
            
        toc_temp = getticks();
        printf("%i %e %e %e %i %i %.3f %.3f %.3f %.3f ms\n",
            e.time,epot,ekin,temp,e.s.nr_swaps,e.s.nr_stalls,
                (double)(toc_temp-tic) * 1000 / CPU_TPS,
                (double)(toc_step-tic) * 1000 / CPU_TPS,
                (double)(toc_bonded-toc_step) * 1000 / CPU_TPS,
                (double)(toc_temp-toc_bonded) * 1000 / CPU_TPS);
        fflush(stdout);
        
        // print some particle data
        /* printf("main: part 1 is at [ %e , %e , %e ].\n",
            e.s.partlist[1]->x[0] + e.s.celllist[1]->origin[0],
            e.s.partlist[1]->x[1] + e.s.celllist[1]->origin[1],
            e.s.partlist[1]->x[2] + e.s.celllist[1]->origin[2]); */
            
        if ( e.time % 100 == 0 ) {
            sprintf( fname , "flexible_%08i.pdb" , e.time ); pdb = fopen( fname , "w" );
            if ( engine_dump_PSF( &e , NULL , pdb , NULL , 0 ) < 0 ) {
                printf("main: engine_dump_PSF failed with engine_err=%i.\n",engine_err);
                errs_dump(stdout);
                return 1;
                }
            fclose(pdb);
            }
    
        } /* Main simulation loop. */
     
    // dump the particle positions, just for the heck of it
    // for ( cid = 0 ; cid < e.s.nr_cells ; cid++ )
    //     for ( pid = 0 ; pid < e.s.cells[cid].count ; pid++ ) {
    //         for ( k = 0 ; k < 3 ; k++ )
    //             x[k] = e.s.cells[cid].origin[k] + e.s.cells[cid].parts[pid].x[k];
    //         printf("%i %e %e %e\n",e.s.cells[cid].parts[pid].id,x[0],x[1],x[2]);
    //         }
    
    psf = fopen( "flexible.psf" , "w" ); pdb = fopen( "flexible.pdb" , "w" );
    if ( engine_dump_PSF( &e , psf , pdb , NULL , 0 ) < 0 ) {
        printf("main: engine_dump_PSF failed with engine_err=%i.\n",engine_err);
        errs_dump(stdout);
        return 1;
        }
    fclose(pdb); fclose(psf);
        
        
    // clean break
    return 0;

    }
