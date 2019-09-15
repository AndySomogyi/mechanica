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

// include some standard headers
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <float.h>
#include <pthread.h>
#include <time.h>

#include "cycle.h"

#include "mdcore_single.h"

/* MPI headers. */
#ifdef WITH_MPI
#include <mpi.h>
#endif

/* FFTW3 headers. */
#ifdef HAVE_FFTW3
#include <complex.h>
#include <fftw3.h>
#endif

/* OpenMP headers. */
#ifdef HAVE_OPENMP
#include <omp.h>
#endif

/* What to do if ENGINE_FLAGS was not defined? */
#ifndef ENGINE_FLAGS
#define ENGINE_FLAGS engine_flag_none
#endif
#ifndef CPU_TPS
#define CPU_TPS 2.67e+9
#endif


int main ( int argc , char *argv[] ) {

    const double origin[3] = { 0.0 , 0.0 , 0.0 };
    // double dim[3] = { 3*3.166 , 3.166 , 3.166 };
    // int nr_mols = 3*1000;
    // const double dim[3] = { 6.332 , 6.332 , 6.332 };
    // const int nr_mols = 8000;
    // const double dim[3] = { 4.0 , 4.0 , 4.0 };
    // const int nr_mols = 2016;
    double dim[3] = { 8.0 , 8.0 , 8.0 };
    int nr_mols = 16128;
    // double dim[3] = { 16.0 , 16.0 , 16.0 };
    // int nr_mols = 129024;
    double Temp = 300.0;
    double cutoff = 1.0;

    double x[3], vtot[3] = { 0.0 , 0.0 , 0.0 };
    double epot, ekin, temp, cellwidth;
    // FPTYPE ee, eff;
    struct engine e;
    struct particle pO, pH;
    struct potential *pot_OO, *pot_OH, *pot_HH;
    // struct potential *pot_ee;
    int i, j, k, cid, pid, nr_runners = 1, nr_steps = 1000;
    int nx, ny, nz;
    double hx, hy, hz;
    double vcom[3], vcom_tot[3], w;

    ticks tic, toc, toc_step, toc_temp;

    double itpms = 1000.0 / CPU_TPS;
    int myrank = 0;
    double L[] = { cutoff , cutoff , cutoff };

    tic = getticks();


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
        cellwidth = cutoff;
    printf("main: cell width set to %22.16e.\n", cellwidth);

    // initialize the engine
    printf("main: initializing the engine... "); fflush(stdout);
    if ( engine_init( &e , origin , dim , L , cutoff , space_periodic_full , 2 , ENGINE_FLAGS | engine_flag_affinity ) != 0 ) {
        printf("main: engine_init failed with engine_err=%i.\n",engine_err);
        errs_dump(stdout);
        return 1;
    }
    printf("done.\n"); fflush(stdout);

#ifdef WITH_CUDA
    if ( engine_cuda_setdevice( &e , 0 ) != 0 ) {
        printf( "main[%i]: engine_cuda_setdevice failed with engine_err=%i.\n" , myrank , engine_err );
        errs_dump(stdout);
        abort();
    }
#endif


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
    if ( ( pot_OH = potential_create_Ewald( 0.1 , 1.0 , -0.35921288 , 3.0 , 1.0e-3 ) ) == NULL ) {
        printf("main: potential_create_Ewald failed with potential_err=%i.\n",potential_err);
        errs_dump(stdout);
        return 1;
    }
    printf("main: constructed OH-potential with %i intervals.\n",pot_OH->n); fflush(stdout);
#ifdef EXPLICIT_POTENTIALS
    pot_OH->flags = potential_flag_Ewald;
    pot_OH->alpha[0] = 0.0;
    pot_OH->alpha[1] = 0.0;
    pot_OH->alpha[2] = -0.35921288;
#endif

    // initialize the H-H potential
    if ( ( pot_HH = potential_create_Ewald( 0.1 , 1.0 , 1.7960644e-1 , 3.0 , 1.0e-3 ) ) == NULL ) {
        printf("main: potential_create_Ewald failed with potential_err=%i.\n",potential_err);
        errs_dump(stdout);
        return 1;
    }
    printf("main: constructed HH-potential with %i intervals.\n",pot_HH->n); fflush(stdout);
#ifdef EXPLICIT_POTENTIALS
    pot_HH->flags = potential_flag_Ewald;
    pot_HH->alpha[0] = 0.0;
    pot_HH->alpha[1] = 0.0;
    pot_HH->alpha[2] = 1.7960644e-1;
#endif

    // initialize the O-O potential
    if ( ( pot_OO = potential_create_LJ126_Ewald( 0.25 , 1.0 , 2.637775819766153e-06 , 2.619222661792581e-03 , 7.1842576e-01 , 3.0 , 1e-3 ) ) == NULL ) {
        printf("main: potential_create_LJ126_Ewald failed with potential_err=%i.\n",potential_err);
        errs_dump(stdout);
        return 1;
    }
    printf("main: constructed OO-potential with %i intervals.\n",pot_OO->n); fflush(stdout);
#ifdef EXPLICIT_POTENTIALS
    pot_OO->flags = potential_flag_LJ126 + potential_flag_Ewald;
    pot_OO->alpha[0] = 2.637775819766153e-06;
    pot_OO->alpha[1] = 2.619222661792581e-03;
    pot_OO->alpha[2] = 7.1842576e-01;
#endif

    // initialize the expl. electrostatic potential
    /* if ( ( pot_ee = potential_create_Ewald( 0.1 , 1.0 , 1.0 , 3.0 , 1.0e-4 ) ) == NULL ) {
        printf("main: potential_create_LJ126_Ewald failed with potential_err=%i.\n",potential_err);
        errs_dump(stdout);
        return 1;
        }
    printf("main: constructed expl. electrostatic potential with %i intervals.\n",pot_ee->n); fflush(stdout);
    if ( engine_setexplepot( &e , pot_ee ) < 0 ) {
        printf("main: engine_setexplepot failed with engine_err=%i.\n",engine_err);
        errs_dump(stdout);
        return 1;
        } */

    /* dump the OO-potential to make sure its ok... 
    for ( i = 0 ; i < 1000 ; i++ ) {
        temp = 0.2 + (double)i/1000 * 0.8;
        potential_eval( pot_OO , temp*temp , &ee , &eff );
        printf("%23.16e %23.16e %23.16e %23.16e %23.16e\n", temp , ee , eff , 
             potential_LJ126(temp,2.637775819766153e-06,2.619222661792581e-03) + 7.1842576e-01*potential_Ewald(temp,3.0) ,
             potential_LJ126_p(temp,2.637775819766153e-06,2.619222661792581e-03) + 7.1842576e-01*potential_Ewald_p(temp,3.0) );
        }
    return 0; */


    /* register the particle types. */
    if ( ( pO.type = engine_addtype( &e , 15.9994 , -0.8476 , "O" , NULL ) ) < 0 ||
            ( pH.type = engine_addtype( &e , 1.00794 , 0.4238 , "H" , NULL ) ) < 0 ) {
        printf("main: call to engine_addtype failed.\n");
        errs_dump(stdout);
        return 1;
    }

    // register these potentials.
    if ( engine_addpot( &e , pot_OO , pO.type , pO.type ) < 0 ||
            engine_addpot( &e , pot_HH , pH.type , pH.type ) < 0 ||
            engine_addpot( &e , pot_OH , pO.type , pH.type ) < 0 ) {
        printf("main: call to engine_addpot failed.\n");
        errs_dump(stdout);
        return 1;
    }

    // set fields for all particles
    srand(6178);
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
        x[0] = 0.05 + i * hx;
        for ( j = 0 ; j < ny ; j++ ) {
            x[1] = 0.05 + j * hy;
            for ( k = 0 ; k < nz && k + nz * ( j + ny * i ) < nr_mols ; k++ ) {
                pO.vid = k + nz * ( j + ny * i );
                pO.id = pO.vid * 3;
                x[2] = 0.05 + k * hz;
                pO.v[0] = ((double)rand()) / RAND_MAX - 0.5;
                pO.v[1] = ((double)rand()) / RAND_MAX - 0.5;
                pO.v[2] = ((double)rand()) / RAND_MAX - 0.5;
                temp = 0.675 / sqrt( pO.v[0]*pO.v[0] + pO.v[1]*pO.v[1] + pO.v[2]*pO.v[2] );
                pO.v[0] *= temp; pO.v[1] *= temp; pO.v[2] *= temp;
                vtot[0] += pO.v[0]; vtot[1] += pO.v[1]; vtot[2] += pO.v[2];
                if ( space_addpart( &(e.s) , &pO , x ) != 0 ) {
                    printf("main: space_addpart failed with space_err=%i.\n",space_err);
                    errs_dump(stdout);
                    return 1;
                }
                x[0] += 0.1;
                pH.vid = pO.vid;
                pH.id = pO.id + 1;
                pH.v[0] = pO.v[0]; pH.v[1] = pO.v[1]; pH.v[2] = pO.v[2];
                if ( space_addpart( &(e.s) , &pH , x ) != 0 ) {
                    printf("main: space_addpart failed with space_err=%i.\n",space_err);
                    errs_dump(stdout);
                    return 1;
                }
                x[0] -= 0.13333;
                x[1] += 0.09428;
                pH.vid = pO.vid;
                pH.id = pO.id + 2;
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
    for ( cid = 0 ; cid < e.s.nr_cells ; cid++ )
        for ( pid = 0 ; pid < e.s.cells[cid].count ; pid++ )
            for ( k = 0 ; k < 3 ; k++ )
                e.s.cells[cid].parts[pid].v[k] -= vtot[k] / nr_mols;
    printf("done.\n"); fflush(stdout);
    printf("main: inserted %i particles.\n", e.s.nr_parts);


    /* Make the constrained bonds and angles for water. */
    for ( k = 0 ; k < nr_mols ; k++ ) {
        if ( engine_rigid_add( &e , 3*k , 3*k+1 , 0.1 ) < 0 ||
                engine_rigid_add( &e , 3*k , 3*k+2 , 0.1 ) < 0 ||
                engine_rigid_add( &e , 3*k+1 , 3*k+2 , 0.163298 ) < 0 ) {
            printf("main: engine_rigid_add failed with engine_err=%i.\n",engine_err);
            errs_dump(stdout);
            abort();
        }
        if ( engine_exclusion_add( &e , 3*k , 3*k+1 ) < 0 ||
                engine_exclusion_add( &e , 3*k , 3*k+2 ) < 0 ||
                engine_exclusion_add( &e , 3*k+1 , 3*k+2 ) < 0 ) {
            printf("main: engine_exclusion_add failed with engine_err=%i.\n",engine_err);
            errs_dump(stdout);
            abort();
        }
    }
    if ( engine_exclusion_shrink( &e ) < 0 ) {
        printf("main: engine_exclusion_shrink failed with engine_err=%i.\n",engine_err);
        errs_dump(stdout);
        return -1;
    }


    // set the time and time-step by hand
    e.time = 0;
    if ( argc > 3 )
        e.dt = atof( argv[3] );
    else
        e.dt = 0.002;
    printf("main: dt set to %f fs.\n", e.dt*1000 );

    toc = getticks();

    printf("main: setup took %.3f ms.\n",(double)(toc-tic) * 1000 / CPU_TPS);

    // did the user specify a number of runners?
#ifdef HAVE_OPENMP
    if ( argc > 1 ) {
        nr_runners = atoi( argv[1] );
        omp_set_num_threads( nr_runners );
    }
#endif

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

        // take a step
        toc_step = getticks();

        // get the total COM-velocities, ekin and epot
        vcom_tot[0] = 0.0; vcom_tot[1] = 0.0; vcom_tot[2] = 0.0;
        ekin = 0.0; epot = e.s.epot;
        for ( j = 0 ; j < nr_mols ; j++ ) {
            for ( k = 0 ; k < 3 ; k++ ) {
                vcom[k] = ( e.s.partlist[j*3]->v[k] * 15.9994 +
                        e.s.partlist[j*3+1]->v[k] * 1.00794 +
                        e.s.partlist[j*3+2]->v[k] * 1.00794 ) / 1.801528e+1;
                vcom_tot[k] += vcom[k];
            }
            ekin += 9.00764 * ( vcom[0]*vcom[0] + vcom[1]*vcom[1] + vcom[2]*vcom[2] );
        }
        for ( k = 0 ; k < 3 ; k++ )
            vcom_tot[k] /= nr_mols * 1.801528e+1;
        // printf("main: vcom_tot is [ %e , %e , %e ].\n",vcom_tot[0],vcom_tot[1],vcom_tot[2]); fflush(stdout);

        // compute the temperature and scaling
        temp = ekin / ( 1.5 * 6.022045E23 * 1.380662E-26 * nr_mols );
        w = sqrt( 1.0 + 0.1 * ( Temp / temp - 1.0 ) );

        // compute the molecular heat
        if ( i < 10000 ) {

            // scale the COM-velocities
            for ( j = 0 ; j < nr_mols ; j++ ) {
                for ( k = 0 ; k < 3 ; k++ ) {
                    vcom[k] = ( e.s.partlist[j*3]->v[k] * 15.9994 +
                            e.s.partlist[j*3+1]->v[k] * 1.00794 +
                            e.s.partlist[j*3+2]->v[k] * 1.00794 ) / 1.801528e+1;
                    vcom[k] -= vcom_tot[k];
                    vcom[k] *= ( w - 1.0 );
                    e.s.partlist[j*3]->v[k] += vcom[k];
                    e.s.partlist[j*3+1]->v[k] += vcom[k];
                    e.s.partlist[j*3+2]->v[k] += vcom[k];
                }
            }

        } // apply molecular thermostat



        toc_step = toc_temp = getticks();

        /* printf("%i %e %e %e %i %i %.3f %.3f %.3f %.3f %.3f %.3f ms\n",
            e.time,epot,ekin,temp,e.s.nr_swaps,e.s.nr_stalls, e.timers[engine_timer_step] * itpms,
            e.timers[engine_timer_nonbond]*itpms, e.timers[engine_timer_bonded]*itpms, e.timers[engine_timer_advance]*itpms, e.timers[engine_timer_rigid]*itpms,
            (toc_temp-toc_step)*itpms ); fflush(stdout); */
        printf("%i %e %e %e %i %i %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f ms\n",
                e.time,epot,ekin,temp,e.s.nr_swaps,e.s.nr_stalls,(toc_temp-tic) * itpms,
                e.timers[engine_timer_nonbond]*itpms, e.timers[engine_timer_bonded]*itpms,
                e.timers[engine_timer_advance]*itpms, e.timers[engine_timer_rigid]*itpms,
                (e.timers[engine_timer_exchange1]+e.timers[engine_timer_exchange2])*itpms,
                e.timers[engine_timer_cuda_load]*itpms, e.timers[engine_timer_cuda_dopairs]*itpms, e.timers[engine_timer_cuda_unload]*itpms,
                (toc_temp - toc_step)*itpms ); fflush(stdout);
        fflush(stdout);

        /* Re-set the timers. */
        if ( engine_timers_reset( &e ) < 0 ) {
            printf("main: engine_timers_reset failed with engine_err=%i.\n",engine_err);
            errs_dump(stdout);
            abort();
        }

        // print some particle data
        // printf("main: part 13322 is at [ %e , %e , %e ].\n",
        //     e.s.partlist[13322]->x[0], e.s.partlist[13322]->x[1], e.s.partlist[13322]->x[2]);

    }

    // dump the particle positions, just for the heck of it
    // for ( cid = 0 ; cid < e.s.nr_cells ; cid++ )
    //     for ( pid = 0 ; pid < e.s.cells[cid].count ; pid++ ) {
    //         for ( k = 0 ; k < 3 ; k++ )
    //             x[k] = e.s.cells[cid].origin[k] + e.s.cells[cid].parts[pid].x[k];
    //         printf("%i %e %e %e\n",e.s.cells[cid].parts[pid].id,x[0],x[1],x[2]);
    //         }


    /* Exit gracefuly. */
    if ( engine_finalize( &e ) < 0 ) {
        printf("main: engine_finalize failed with engine_err=%i.\n",engine_err);
        errs_dump(stdout);
        abort();
    }
    fflush(stdout);
    printf( "main: exiting.\n" );
    return 0;

}
