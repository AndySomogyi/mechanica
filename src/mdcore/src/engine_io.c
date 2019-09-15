/*******************************************************************************
 * This file is part of mdcore.
 * Coypright (c) 2010 Pedro Gonnet (pedro.gonnet@durham.ac.uk)
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


/* include some standard header files */
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <math.h>
#include <float.h>
#include <string.h>

/* Include conditional headers. */
#include "config.h"
#ifdef WITH_MPI
    #include <mpi.h>
#endif
#ifdef HAVE_OPENMP
    #include <omp.h>
#endif

/* include local headers */
#include "cycle.h"
#include "errs.h"
#include "fptype.h"
#include "lock.h"
#include <particle.h>
#include <space_cell.h>
#include "space.h"
#include "potential.h"
#include "runner.h"
#include "bond.h"
#include "rigid.h"
#include "angle.h"
#include "dihedral.h"
#include "exclusion.h"
#include "reader.h"
#include "engine.h"


/* the error macro. */
#define error(id)				( engine_err = errs_register( id , engine_err_msg[-(id)] , __LINE__ , __FUNCTION__ , __FILE__ ) )



/**
 * @brief Read the potentials from a XPLOR parameter file.
 *
 * @param e The #engine.
 * @param xplor The open XPLOR parameter file.
 * @param kappa The PME screening width.
 * @param tol The absolute tolerance for interpolation.
 * @param rigidH Convert all bonds over a type starting with 'H'
 *      to a rigid constraint.
 *
 * If @c kappa is zero, truncated Coulomb electrostatic interactions are
 * assumed. If @c kappa is less than zero, no electrostatic interactions
 * are computed.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */

int engine_read_xplor ( struct engine *e , int xplor , double kappa , double tol , int rigidH ) {

    struct reader r;
    char buff[100], type1[100], type2[100], type3[100], type4[100], *endptr;
    int tid, tjd, wc[4];
    int res, j, k, jj, kk, n, *ind1, *ind2, nr_ind1, nr_ind2, potid;
    double K, Kr0, r0, r2, r6, A, B, q, al, ar, am, vm;
    struct potential *p;
    
    /* Check inputs. */
    if ( e == NULL )
        return error(engine_err_null);
        
    /* Allocate some local memory for the index arrays. */
    if ( ( ind1 = (int *)alloca( sizeof(int) * e->nr_types ) ) == NULL ||
         ( ind2 = (int *)alloca( sizeof(int) * e->nr_types ) ) == NULL )
        return error(engine_err_malloc);
        
    /* Init the reader with the XPLOR file. */
    if ( reader_init( &r , xplor , NULL , "!{" , "\n" , engine_readbuff ) < 0 )
        return error(engine_err_reader);
        
    /* Main loop. */
    while ( !( r.flags & reader_flag_eof ) ) {
    
        /* Get the first token */
        if ( ( res = reader_gettoken( &r , buff , 100 ) ) == reader_err_eof )
            break;
        else if ( res < 0 )
            return error(engine_err_reader);

        /* Did we get a bond? */
        if ( strncasecmp( buff , "BOND" , 4 ) == 0 ) {
    
            /* Get the atom types. */
            if ( reader_gettoken( &r , type1 , 100 ) < 0 )
                return error(engine_err_reader);
            if ( reader_gettoken( &r , type2 , 100 ) < 0 )
                return error(engine_err_reader);

            /* Get the parameters K and r0. */
            if ( reader_gettoken( &r , buff , 100 ) < 0 )
                return error(engine_err_reader);
            K = strtod( buff , &endptr );
            if ( *endptr != 0 )
                return error(engine_err_cpf);
            if ( reader_gettoken( &r , buff , 100 ) < 0 )
                return error(engine_err_reader);
            r0 = strtod( buff , &endptr );
            if ( *endptr != 0 )
                return error(engine_err_cpf);

            /* Is this a rigid bond (and do we care)? */  
            if ( rigidH && ( type1[0] == 'H' || type2[0] == 'H' ) ) {

                /* Loop over all bonds... */
                for ( k = 0 ; k < e->nr_bonds ; k++ ) {

                    /* Does this bond match the types? */
                    if ( ( strcmp( e->types[e->s.partlist[e->bonds[k].i]->type].name , type1 ) == 0 &&
                           strcmp( e->types[e->s.partlist[e->bonds[k].j]->type].name , type2 ) == 0 ) ||
                         ( strcmp( e->types[e->s.partlist[e->bonds[k].i]->type].name , type2 ) == 0 &&
                           strcmp( e->types[e->s.partlist[e->bonds[k].j]->type].name , type1 ) == 0 ) ) {

                        /* Register as a constraint. */
                        if ( engine_rigid_add( e , e->bonds[k].i , e->bonds[k].j , 0.1*r0 ) < 0 )
                            return error(engine_err);

                        /* Remove this bond. */
                        e->nr_bonds -= 1;
                        e->bonds[k] = e->bonds[e->nr_bonds];
                        k -= 1;

                        }

                    } /* Loop over all bonds. */

                }

            /* Otherwise... */
            else {

                /* Are type1 and type2 the same? */
                if ( strcmp( type1 , type2 ) == 0 ) {

                    /* Fill the ind1 array. */
                    for ( nr_ind1 = 0 , k = 0 ; k < e->nr_types ; k++ )
                        if ( strcmp( type1 , e->types[k].name ) == 0 ) {
                            ind1[nr_ind1] = k;
                            nr_ind1 += 1;
                            }

                    /* Are there any indices? */
                    if ( nr_ind1 > 0 ) {

                        /* Create the harmonic potential. */
                        if ( ( p = potential_create_harmonic( 0.05*r0 , 0.15*r0 , 418.4*K , 0.1*r0 , tol ) ) == NULL )
                            return error(engine_err_potential);

                        /* Loop over the types and add the potential. */
                        for ( tid = 0 ; tid < nr_ind1 ; tid++ )
                            for ( tjd = tid ; tjd < nr_ind1 ; tjd++ )
                                if ( engine_bond_addpot( e , p , ind1[tid] , ind1[tjd] ) < 0 )
                                    return error(engine_err);

                        }

                    }
                /* Otherwise... */
                else {

                    /* Fill the ind1 and ind2 arrays. */
                    for ( nr_ind1 = 0 , nr_ind2 = 0 , k = 0 ; k < e->nr_types ; k++ ) {
                        if ( strcmp( type1 , e->types[k].name ) == 0 ) {
                            ind1[nr_ind1] = k;
                            nr_ind1 += 1;
                            }
                        else if ( strcmp( type2 , e->types[k].name ) == 0 ) {
                            ind2[nr_ind2] = k;
                            nr_ind2 += 1;
                            }
                        }

                    /* Are there any indices? */
                    if ( nr_ind1 > 0 && nr_ind2 > 0 ) {

                        /* Create the harmonic potential. */
                        if ( ( p = potential_create_harmonic( 0.05*r0 , 0.15*r0 , 418.4*K , 0.1*r0 , tol ) ) == NULL )
                            return error(engine_err_potential);

                        /* Loop over the types and add the potential. */
                        for ( tid = 0 ; tid < nr_ind1 ; tid++ )
                            for ( tjd = 0 ; tjd < nr_ind2 ; tjd++ )
                                if ( engine_bond_addpot( e , p , ind1[tid] , ind2[tjd] ) < 0 )
                                    return error(engine_err);

                        }

                    }

                }
                
            } /* Is it a bond? */
        
        /* Is it an angle? */    
        else if ( strncasecmp( buff , "ANGL" , 4 ) == 0 ) {
        
            /* Get the atom types. */
            if ( reader_gettoken( &r , type1 , 100 ) < 0 )
                return error(engine_err_reader);
            if ( reader_gettoken( &r , type2 , 100 ) < 0 )
                return error(engine_err_reader);
            if ( reader_gettoken( &r , type3 , 100 ) < 0 )
                return error(engine_err_reader);

            /* Check if these types even exist. */
            if ( engine_gettype( e , type1 ) < 0 && 
                 engine_gettype( e , type2 ) < 0 &&
                 engine_gettype( e , type3 ) < 0 ) {
                if ( reader_skipline( &r ) < 0 )
                    return error(engine_err_reader);
                continue;
                }

            /* Get the parameters K and r0. */
            if ( reader_gettoken( &r , buff , 100 ) < 0 )
                return error(engine_err_reader);
            K = strtod( buff , &endptr );
            if ( *endptr != 0 )
                return error(engine_err_cpf);
            if ( reader_gettoken( &r , buff , 100 ) < 0 )
                return error(engine_err_reader);
            r0 = strtod( buff , &endptr );
            if ( *endptr != 0 )
                return error(engine_err_cpf);

            /* Run through the angle list and create the potential if necessary. */
            potid = -1;
            for ( k = 0 ; k < e->nr_angles ; k++ ) {

                /* Does this angle match the types? */
                if ( ( strcmp( e->types[e->s.partlist[e->angles[k].i]->type].name , type1 ) == 0 &&
                       strcmp( e->types[e->s.partlist[e->angles[k].j]->type].name , type2 ) == 0 &&
                       strcmp( e->types[e->s.partlist[e->angles[k].k]->type].name , type3 ) == 0 ) ||
                     ( strcmp( e->types[e->s.partlist[e->angles[k].i]->type].name , type3 ) == 0 &&
                       strcmp( e->types[e->s.partlist[e->angles[k].j]->type].name , type2 ) == 0 &&
                       strcmp( e->types[e->s.partlist[e->angles[k].k]->type].name , type1 ) == 0 ) ) {

                    /* Do we need to create the potential? */
                    if ( potid < 0 ) {
                        if ( ( p = potential_create_harmonic_angle( M_PI/180*(r0-45) , M_PI/180*(r0+45) , 4.184*K , M_PI/180*r0 , tol ) ) == NULL )
                            return error(engine_err_potential);
                        if ( ( potid = engine_angle_addpot( e , p ) ) < 0 )
                            return error(engine_err);
                        /* printf( "engine_read_cpf: generated potential for angle %s %s %s with %i intervals.\n" ,
                            type1 , type2 , type3 , e->p_angle[potid]->n ); */
                        }

                    /* Add the potential to the angle. */
                    e->angles[k].pid = potid;

                    }

                }
            
            } /* Is it an angle? */
            
        /* Perhaps a propper dihedral? */
        else if ( strncasecmp( buff , "DIHE" , 4 ) == 0 ) {
        
            /* Get the atom types. */
            if ( reader_gettoken( &r , type1 , 100 ) < 0 )
                return error(engine_err_reader);
            if ( reader_gettoken( &r , type2 , 100 ) < 0 )
                return error(engine_err_reader);
            if ( reader_gettoken( &r , type3 , 100 ) < 0 )
                return error(engine_err_reader);
            if ( reader_gettoken( &r , type4 , 100 ) < 0 )
                return error(engine_err_reader);

            /* Check for wildcards. */
            wc[0] = ( strcmp( type1 , "X" ) == 0 );
            wc[1] = ( strcmp( type2 , "X" ) == 0 );
            wc[2] = ( strcmp( type3 , "X" ) == 0 );
            wc[3] = ( strcmp( type4 , "X" ) == 0 );

            /* Check if these types even exist. */
            if ( ( wc[0] || engine_gettype( e , type1 ) < 0 ) && 
                 ( wc[1] || engine_gettype( e , type2 ) < 0 ) &&
                 ( wc[2] || engine_gettype( e , type3 ) < 0 ) &&
                 ( wc[3] || engine_gettype( e , type4 ) < 0 ) ) {
                if ( reader_skipline( &r ) < 0 )
                    return error(engine_err_reader);
                continue;
                }

            /* Get the parameters K and r0. */
            if ( reader_gettoken( &r , buff , 100 ) < 0 )
                return error(engine_err_reader);
            K = strtod( buff , &endptr );
            if ( *endptr != 0 ) {
                printf( "engine_read_xplor: failed to parse double on line %i, col %i.\n" , r.line , r.col );
                return error(engine_err_cpf);
                }
            if ( reader_gettoken( &r , buff , 100 ) < 0 )
                return error(engine_err_reader);
            n = strtol( buff , &endptr , 0 );
            if ( *endptr != 0 ) {
                printf( "engine_read_xplor: failed to parse int on line %i, col %i.\n" , r.line , r.col );
                return error(engine_err_cpf);
                }
            if ( reader_gettoken( &r , buff , 100 ) < 0 )
                return error(engine_err_reader);
            r0 = strtod( buff , &endptr );
            if ( *endptr != 0 ) {
                printf( "engine_read_xplor: failed to parse double on line %i, col %i.\n" , r.line , r.col );
                return error(engine_err_cpf);
                }

            /* Run through the dihedral list and create the potential if necessary. */
            potid = -1;
            for ( k = 0 ; k < e->nr_dihedrals ; k++ ) {

                /* Does this dihedral match the types? */
                if ( ( e->dihedrals[k].pid == -1 ) &&
                     ( ( ( wc[0] || strcmp( e->types[e->s.partlist[e->dihedrals[k].i]->type].name , type1 ) == 0 ) &&
                         ( wc[1] || strcmp( e->types[e->s.partlist[e->dihedrals[k].j]->type].name , type2 ) == 0 ) &&
                         ( wc[2] || strcmp( e->types[e->s.partlist[e->dihedrals[k].k]->type].name , type3 ) == 0 ) &&
                         ( wc[3] || strcmp( e->types[e->s.partlist[e->dihedrals[k].l]->type].name , type4 ) == 0 ) ) ||
                       ( ( wc[3] || strcmp( e->types[e->s.partlist[e->dihedrals[k].i]->type].name , type4 ) == 0 ) &&
                         ( wc[2] || strcmp( e->types[e->s.partlist[e->dihedrals[k].j]->type].name , type3 ) == 0 ) &&
                         ( wc[1] || strcmp( e->types[e->s.partlist[e->dihedrals[k].k]->type].name , type2 ) == 0 ) &&
                         ( wc[0] || strcmp( e->types[e->s.partlist[e->dihedrals[k].l]->type].name , type1 ) == 0 ) ) ) ) {

                    /* Do we need to create the potential? */
                    if ( potid < 0 ) {
                        if ( ( p = potential_create_harmonic_dihedral( 4.184*K , n , M_PI/180*r0 , tol ) ) == NULL )
                            return error(engine_err_potential);
                        if ( ( potid = engine_dihedral_addpot( e , p ) ) < 0 )
                            return error(engine_err);
                        /* printf( "engine_read_cpf: generated potential for dihedral %s %s %s %s in [%e,%e] with %i intervals.\n" ,
                            type1 , type2 , type3 , type4 , e->p_dihedral[potid]->a , e->p_dihedral[potid]->b , e->p_dihedral[potid]->n ); */
                        }

                    /* Add the potential to the dihedral. */
                    e->dihedrals[k].pid = potid;

                    }

                }
            
            } /* Dihedral? */
            
        /* Or an improper dihedral instead? */
        else if ( strncasecmp( buff , "IMPR" , 4 ) == 0 ) {
        
            /* Get the atom types. */
            if ( reader_gettoken( &r , type1 , 100 ) < 0 )
                return error(engine_err_reader);
            if ( reader_gettoken( &r , type2 , 100 ) < 0 )
                return error(engine_err_reader);
            if ( reader_gettoken( &r , type3 , 100 ) < 0 )
                return error(engine_err_reader);
            if ( reader_gettoken( &r , type4 , 100 ) < 0 )
                return error(engine_err_reader);

            /* Check for wildcards. */
            wc[0] = ( strcmp( type1 , "X" ) == 0 );
            wc[1] = ( strcmp( type2 , "X" ) == 0 );
            wc[2] = ( strcmp( type3 , "X" ) == 0 );
            wc[3] = ( strcmp( type4 , "X" ) == 0 );

            /* Check if these types even exist. */
            if ( ( wc[0] || engine_gettype( e , type1 ) < 0 ) && 
                 ( wc[1] || engine_gettype( e , type2 ) < 0 ) &&
                 ( wc[2] || engine_gettype( e , type3 ) < 0 ) &&
                 ( wc[3] || engine_gettype( e , type4 ) < 0 ) ) {
                if ( reader_skipline( &r ) < 0 )
                    return error(engine_err_reader);
                continue;
                }

            /* Get the parameters K and r0. */
            if ( reader_gettoken( &r , buff , 100 ) < 0 )
                return error(engine_err_reader);
            K = strtod( buff , &endptr );
            if ( *endptr != 0 )
                return error(engine_err_cpf);
            if ( reader_skiptoken( &r ) < 0 )
                return error(engine_err_reader);
            if ( reader_gettoken( &r , buff , 100 ) < 0 )
                return error(engine_err_reader);
            r0 = strtod( buff , &endptr );
            if ( *endptr != 0 )
                return error(engine_err_cpf);

            /* Run through the dihedral list and create the potential if necessary. */
            potid = -1;
            for ( k = 0 ; k < e->nr_dihedrals ; k++ ) {

                /* Does this dihedral match the types? */
                if ( ( e->dihedrals[k].pid == -2 ) &&
                     ( ( ( wc[0] || strcmp( e->types[e->s.partlist[e->dihedrals[k].i]->type].name , type1 ) == 0 ) &&
                         ( wc[1] || strcmp( e->types[e->s.partlist[e->dihedrals[k].j]->type].name , type2 ) == 0 ) &&
                         ( wc[2] || strcmp( e->types[e->s.partlist[e->dihedrals[k].k]->type].name , type3 ) == 0 ) &&
                         ( wc[3] || strcmp( e->types[e->s.partlist[e->dihedrals[k].l]->type].name , type4 ) == 0 ) ) ||
                       ( ( wc[3] || strcmp( e->types[e->s.partlist[e->dihedrals[k].i]->type].name , type4 ) == 0 ) &&
                         ( wc[2] || strcmp( e->types[e->s.partlist[e->dihedrals[k].j]->type].name , type3 ) == 0 ) &&
                         ( wc[1] || strcmp( e->types[e->s.partlist[e->dihedrals[k].k]->type].name , type2 ) == 0 ) &&
                         ( wc[0] || strcmp( e->types[e->s.partlist[e->dihedrals[k].l]->type].name , type1 ) == 0 ) ) ) ) {

                    /* Do we need to create the potential? */
                    if ( potid < 0 ) {
                        if ( ( p = potential_create_harmonic_angle( M_PI/180*(r0-45) , M_PI/180*(r0+45) , 4.184*K , M_PI/180*r0 , tol ) ) == NULL )
                            return error(engine_err_potential);
                        if ( ( potid = engine_dihedral_addpot( e , p ) ) < 0 )
                            return error(engine_err);
                        /* printf( "engine_read_cpf: generated potential for imp. dihedral %s %s %s %s with %i intervals.\n" ,
                            type1 , type2 , type3 , type4 , e->p_dihedral[potid]->n ); */
                        }

                    /* Add the potential to the dihedral. */
                    e->dihedrals[k].pid = potid;

                    }

                }
            
            } /* Improper dihedral? */
            
        /* Well then maybe a non-bonded interaction... */
        else if ( strncasecmp( buff , "NONB" , 4 ) == 0 ) {
        
            /* Get the atom type. */
            if ( reader_gettoken( &r , type1 , 100 ) < 0 )
                return error(engine_err_reader);

            /* Get the next two parameters. */
            if ( reader_gettoken( &r , buff , 100 ) < 0 )
                return error(engine_err_reader);
            K = strtod( buff , &endptr );
            if ( *endptr != 0 )
                return error(engine_err_cpf);
            if ( reader_gettoken( &r , buff , 100 ) < 0 )
                return error(engine_err_reader);
            r0 = strtod( buff , &endptr );
            if ( *endptr != 0 )
                return error(engine_err_cpf);

            /* Run through the types and store the parameters for each match. */
            for ( k = 0 ; k < e->nr_types ; k++ )
                if ( strcmp( e->types[k].name , type1 ) == 0 ) {
                    e->types[k].eps = 4.184 * K;
                    e->types[k].rmin = 0.05 * r0;
                    }
                
            } /* non-bonded iteraction. */
            
        /* Otherwise, do, well, nothing. */
        else {
        
            }
            
        /* Skip the rest of the line. */
        if ( reader_skipline( &r ) < 0 )
            return error(engine_err_reader);
    
        } /* Main reading loop. */
        
    /* Close the reader. */
    reader_close( &r );
        
                        
    /* Loop over all the type pairs and construct the non-bonded potentials. */
    for ( j = 0 ; j < e->nr_types ; j++ )
        for ( k = j ; k < e->nr_types ; k++ ) {
            
            /* Has a potential been specified for this case? */
            if ( ( e->types[j].eps == 0.0 || e->types[k].eps == 0.0 ) &&
                 ( kappa < 0.0 || e->types[j].charge == 0.0 || e->types[k].charge == 0.0 ) )
                continue;
                
            /* Has this potential already been specified? */
            if ( kappa < 0.0 ) {
                for ( jj = 0 ; jj < j && ( e->types[jj].eps != e->types[j].eps || e->types[jj].rmin != e->types[j].rmin ) ; jj++ );
                for ( kk = 0 ; kk < k && ( e->types[kk].eps != e->types[k].eps || e->types[kk].rmin != e->types[k].rmin ) ; kk++ );
                }
            else {
                for ( jj = 0 ; jj < j && ( e->types[jj].eps != e->types[j].eps || e->types[jj].rmin != e->types[j].rmin || e->types[jj].charge != e->types[j].charge ) ; jj++ );
                for ( kk = 0 ; kk < k && ( e->types[kk].eps != e->types[k].eps || e->types[kk].rmin != e->types[k].rmin || e->types[kk].charge != e->types[k].charge ) ; kk++ );
                }
            if ( jj < j && kk < k ) {
                if ( e->p[ jj + e->max_type*kk ] != NULL && engine_addpot( e , e->p[ jj + e->max_type*kk ] , j , k ) < 0 )
                    return error(engine_err);
                continue;
                }
                    
            /* Construct the common LJ parameters. */
            K = sqrt( e->types[j].eps * e->types[k].eps );
            r0 = e->types[j].rmin + e->types[k].rmin;
            r2 = r0*r0; r6 = r2*r2*r2;
            A = K*r6*r6; B = 2*K*r6;
            q = e->types[j].charge*e->types[k].charge;
                
            /* Construct the potential. */
            /* printf( "engine_read_cpf: creating %s-%s potential with A=%e B=%e q=%e.\n" ,
                e->types[j].name , e->types[k].name , 
                K*r6*r6 , K*2*r6 , e->types[j].charge*e->types[k].charge ); */
            if ( K == 0.0 ) {
                if ( q != 0.0 && kappa >= 0.0 ) {
                    if ( kappa > 0.0 ) {
                        if ( ( p = potential_create_Ewald( 0.1 , e->s.cutoff , q , kappa , tol ) ) == NULL )
                            return error(engine_err_potential);
                        }
                    else {
                        if ( ( p = potential_create_Coulomb( 0.1 , e->s.cutoff , q , tol ) ) == NULL )
                            return error(engine_err_potential);
                        }
                    }
                else
                    p = NULL;
                }
            if ( kappa < 0.0 ) {
                al = r0/2;
                ar = r0;
                Kr0 = fabs( potential_LJ126( r0 , A , B ) );
                while ( ar-al > 1e-5 ) {
                    am = 0.5*(al + ar); vm = potential_LJ126( am , A , B );
                    if ( fabs(vm) < engine_maxKcutoff*Kr0 )
                        ar = am;
                    else
                        al = am;
                    }
                if ( ( p = potential_create_LJ126( al , e->s.cutoff , A , B , tol ) ) == NULL ) {
                    printf( "engine_read_xplor: failed to create %s-%s potential with A=%e B=%e on [%e,%e].\n" ,
                    e->types[j].name , e->types[k].name , 
                    A , B , al , e->s.cutoff );
                    return error(engine_err_potential);
                    }
                }
            else if ( kappa == 0.0 ) {
                al = r0/2;
                ar = r0;
                Kr0 = fabs( potential_LJ126( r0 , A , B ) + potential_escale*q/r0 );
                while ( ar-al > 1e-5 ) {
                    am = 0.5*(al + ar); vm = potential_LJ126( am , A , B ) + potential_escale*q/am;
                    if ( fabs(vm) < engine_maxKcutoff*Kr0 )
                        ar = am;
                    else
                        al = am;
                    }
                if ( ( p = potential_create_LJ126_Coulomb( al , e->s.cutoff , A , B , q , tol ) ) == NULL ) {
                    printf( "engine_read_xplor: failed to create %s-%s potential with A=%e B=%e q=%e on [%e,%e].\n" ,
                    e->types[j].name , e->types[k].name , 
                    A ,B , q ,
                    al , e->s.cutoff );
                    return error(engine_err_potential);
                    }
                }
            else  {
                al = r0/2;
                ar = r0;
                Kr0 = fabs( potential_LJ126( r0 , A , B ) + q*potential_Ewald( r0, kappa ) );
                while ( ar-al > 1e-5 ) {
                    am = 0.5*(al + ar); vm = potential_LJ126( am , A , B ) + q*potential_Ewald( am , kappa );
                    if ( fabs(vm) < engine_maxKcutoff*Kr0 )
                        ar = am;
                    else
                        al = am;
                    }
                if ( ( p = potential_create_LJ126_Ewald( al , e->s.cutoff , A , B , q , kappa , tol ) ) == NULL ) {
                    printf( "engine_read_xplor: failed to create %s-%s potential with A=%e B=%e q=%e on [%e,%e].\n" ,
                    e->types[j].name , e->types[k].name , 
                    A , B , q ,
                    al , e->s.cutoff );
                    return error(engine_err_potential);
                    }
                }
                
            /* Register it with the local authorities. */
            if ( p != NULL && engine_addpot( e , p , j , k ) < 0 )
                return error(engine_err);
                
            }
        
    /* It's been a hard day's night. */
    return engine_err_ok;
        
    }


/**
 * @brief Read the potentials from a CHARMM parameter file.
 *
 * @param e The #engine.
 * @param cpf The open CHARMM parameter file.
 * @param kappa The PME screening width.
 * @param tol The absolute tolerance for interpolation.
 * @param rigidH Convert all bonds over a type starting with 'H'
 *      to a rigid constraint.
 *
 * If @c kappa is zero, truncated Coulomb electrostatic interactions are
 * assumed. If @c kappa is less than zero, no electrostatic interactions
 * are computed.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */

int engine_read_cpf ( struct engine *e , int cpf , double kappa , double tol , int rigidH ) {

    struct reader r;
    char buff[100], type1[100], type2[100], type3[100], type4[100], *endptr;
    int tid, tjd, wc[4];
    int j, k, jj, kk, n, *ind1, *ind2, nr_ind1, nr_ind2, potid;
    double K, Kr0, r0, r2, r6;
    double al, ar, am, vm, A, B, q;
    struct potential *p;
    
    /* Check inputs. */
    if ( e == NULL )
        return error(engine_err_null);
        
    /* Allocate some local memory for the index arrays. */
    if ( ( ind1 = (int *)alloca( sizeof(int) * e->nr_types ) ) == NULL ||
         ( ind2 = (int *)alloca( sizeof(int) * e->nr_types ) ) == NULL )
        return error(engine_err_malloc);
        
    /* Init the reader with the PSF file. */
    if ( reader_init( &r , cpf , NULL , "!" , "\n" , engine_readbuff ) < 0 )
        return error(engine_err_reader);
        
    /* Skip all lines starting with a "*". */
    while ( r.c == '*' )
        if ( reader_skipline( &r ) < 0 )
            return error(engine_err_reader);
            
    
    /* We should now have the keword starting with "BOND". */
    if ( reader_gettoken( &r , buff , 100 ) < 0 )
        return error(engine_err_reader);
    if ( strncmp( buff , "BOND" , 4 ) != 0 )
        return error(engine_err_cpf);
        
    /* Bond-reading loop. */
    while ( 1 ) {
    
        /* Get a token. */
        if ( reader_gettoken( &r , type1 , 100 ) < 0 )
            return error(engine_err_reader);
            
        /* If it's the start of the "ANGLe" section, break. */
        if ( strncmp( type1 , "ANGL" , 4 ) == 0 )
            break;
            
        /* Otherwise, get the next token, e.g. the second type. */
        if ( reader_gettoken( &r , type2 , 100 ) < 0 )
            return error(engine_err_reader);
            
        /* Get the parameters K and r0. */
        if ( reader_gettoken( &r , buff , 100 ) < 0 )
            return error(engine_err_reader);
        K = strtod( buff , &endptr );
        if ( *endptr != 0 )
            return error(engine_err_cpf);
        if ( reader_gettoken( &r , buff , 100 ) < 0 )
            return error(engine_err_reader);
        r0 = strtod( buff , &endptr );
        if ( *endptr != 0 )
            return error(engine_err_cpf);
          
        /* Is this a rigid bond (and do we care)? */  
        // if ( rigidH && ( ( type1[0] == 'H' && type1[1] == 'T' ) || ( type2[0] == 'H' && type2[1] == 'T' ) ) ) {
        if ( rigidH && ( type1[0] == 'H' || type2[0] == 'H' ) ) {
        
            /* Loop over all bonds... */
            for ( k = 0 ; k < e->nr_bonds ; k++ ) {
            
                /* Does this bond match the types? */
                if ( ( strcmp( e->types[e->s.partlist[e->bonds[k].i]->type].name , type1 ) == 0 &&
                       strcmp( e->types[e->s.partlist[e->bonds[k].j]->type].name , type2 ) == 0 ) ||
                     ( strcmp( e->types[e->s.partlist[e->bonds[k].i]->type].name , type2 ) == 0 &&
                       strcmp( e->types[e->s.partlist[e->bonds[k].j]->type].name , type1 ) == 0 ) ) {
                       
                    /* Register as a constraint. */
                    if ( engine_rigid_add( e , e->bonds[k].i , e->bonds[k].j , 0.1*r0 ) < 0 )
                        return error(engine_err);
                        
                    /* Remove this bond. */
                    e->nr_bonds -= 1;
                    e->bonds[k] = e->bonds[e->nr_bonds];
                    k -= 1;
                    
                    }
            
                } /* Loop over all bonds. */
        
            }
            
        /* Otherwise... */
        else {
            
            /* Are type1 and type2 the same? */
            if ( strcmp( type1 , type2 ) == 0 ) {

                /* Fill the ind1 array. */
                for ( nr_ind1 = 0 , k = 0 ; k < e->nr_types ; k++ )
                    if ( strcmp( type1 , e->types[k].name ) == 0 ) {
                        ind1[nr_ind1] = k;
                        nr_ind1 += 1;
                        }

                /* Are there any indices? */
                if ( nr_ind1 > 0 ) {

                    /* Create the harmonic potential. */
                    if ( ( p = potential_create_harmonic( 0.05*r0 , 0.15*r0 , 418.4*K , 0.1*r0 , tol ) ) == NULL )
                        return error(engine_err_potential);

                    /* Loop over the types and add the potential. */
                    for ( tid = 0 ; tid < nr_ind1 ; tid++ )
                        for ( tjd = tid ; tjd < nr_ind1 ; tjd++ )
                            if ( engine_bond_addpot( e , p , ind1[tid] , ind1[tjd] ) < 0 )
                                return error(engine_err);

                    }

                }
            /* Otherwise... */
            else {

                /* Fill the ind1 and ind2 arrays. */
                for ( nr_ind1 = 0 , nr_ind2 = 0 , k = 0 ; k < e->nr_types ; k++ ) {
                    if ( strcmp( type1 , e->types[k].name ) == 0 ) {
                        ind1[nr_ind1] = k;
                        nr_ind1 += 1;
                        }
                    else if ( strcmp( type2 , e->types[k].name ) == 0 ) {
                        ind2[nr_ind2] = k;
                        nr_ind2 += 1;
                        }
                    }

                /* Are there any indices? */
                if ( nr_ind1 > 0 && nr_ind2 > 0 ) {

                    /* Create the harmonic potential. */
                    if ( ( p = potential_create_harmonic( 0.05*r0 , 0.15*r0 , 418.4*K , 0.1*r0 , tol ) ) == NULL )
                        return error(engine_err_potential);

                    /* Loop over the types and add the potential. */
                    for ( tid = 0 ; tid < nr_ind1 ; tid++ )
                        for ( tjd = 0 ; tjd < nr_ind2 ; tjd++ )
                            if ( engine_bond_addpot( e , p , ind1[tid] , ind2[tjd] ) < 0 )
                                return error(engine_err);

                    }
                    
                }
                
            }
            
        /* Skip the rest of the line. */
        if ( reader_skipline( &r ) < 0 )
            return error(engine_err_reader);
    
        } /* bond-reading loop. */
        
        
    /* Skip the rest of the "ANGLe" line. */
    if ( reader_skipline( &r ) < 0 )
        return error(engine_err_reader);
        
    /* Main angle loop. */
    while ( 1 ) {
    
        /* Get a token. */
        if ( reader_gettoken( &r , type1 , 100 ) < 0 )
            return error(engine_err_reader);
            
        /* If it's the start of the "DIHEdral" section, break. */
        if ( strncmp( type1 , "DIHE" , 4 ) == 0 )
            break;
            
        /* Otherwise, get the next two tokens, e.g. the second and third type. */
        if ( reader_gettoken( &r , type2 , 100 ) < 0 )
            return error(engine_err_reader);
        if ( reader_gettoken( &r , type3 , 100 ) < 0 )
            return error(engine_err_reader);
            
        /* Check if these types even exist. */
        if ( engine_gettype( e , type1 ) < 0 && 
             engine_gettype( e , type2 ) < 0 &&
             engine_gettype( e , type3 ) < 0 ) {
            if ( reader_skipline( &r ) < 0 )
                return error(engine_err_reader);
            continue;
            }
            
        /* Get the parameters K and r0. */
        if ( reader_gettoken( &r , buff , 100 ) < 0 )
            return error(engine_err_reader);
        K = strtod( buff , &endptr );
        if ( *endptr != 0 )
            return error(engine_err_cpf);
        if ( reader_gettoken( &r , buff , 100 ) < 0 )
            return error(engine_err_reader);
        r0 = strtod( buff , &endptr );
        if ( *endptr != 0 )
            return error(engine_err_cpf);
            
        /* Run through the angle list and create the potential if necessary. */
        potid = -1;
        for ( k = 0 ; k < e->nr_angles ; k++ ) {
        
            /* Does this angle match the types? */
            if ( ( strcmp( e->types[e->s.partlist[e->angles[k].i]->type].name , type1 ) == 0 &&
                   strcmp( e->types[e->s.partlist[e->angles[k].j]->type].name , type2 ) == 0 &&
                   strcmp( e->types[e->s.partlist[e->angles[k].k]->type].name , type3 ) == 0 ) ||
                 ( strcmp( e->types[e->s.partlist[e->angles[k].i]->type].name , type3 ) == 0 &&
                   strcmp( e->types[e->s.partlist[e->angles[k].j]->type].name , type2 ) == 0 &&
                   strcmp( e->types[e->s.partlist[e->angles[k].k]->type].name , type1 ) == 0 ) ) {
                
                /* Do we need to create the potential? */
                if ( potid < 0 ) {
                    if ( ( p = potential_create_harmonic_angle( M_PI/180*(r0-30) , M_PI/180*(r0+30) , 4.184*K , M_PI/180*r0 , tol ) ) == NULL )
                        return error(engine_err_potential);
                    if ( ( potid = engine_angle_addpot( e , p ) ) < 0 )
                        return error(engine_err);
                    /* printf( "engine_read_cpf: generated potential for angle %s %s %s with %i intervals.\n" ,
                        type1 , type2 , type3 , e->p_angle[potid]->n ); */
                    }
                
                /* Add the potential to the angle. */
                e->angles[k].pid = potid;
                
                }
        
            }
            
        /* Skip the rest of this line. */
        if ( reader_skipline( &r ) < 0 )
            return error(engine_err_reader);
            
        } /* angle loop. */
        
    
    /* Skip the rest of the "DIHEdral" line. */
    if ( reader_skipline( &r ) < 0 )
        return error(engine_err_reader);
        
    /* Main dihedral loop. */
    while ( 1 ) {
    
        /* Get a token. */
        if ( reader_gettoken( &r , type1 , 100 ) < 0 )
            return error(engine_err_reader);
            
        /* If it's the start of the "IMPRoper" section, break. */
        if ( strncmp( type1 , "IMPR" , 4 ) == 0 )
            break;
            
        /* Otherwise, get the next three tokens, e.g. the second, third and fouth type. */
        if ( reader_gettoken( &r , type2 , 100 ) < 0 )
            return error(engine_err_reader);
        if ( reader_gettoken( &r , type3 , 100 ) < 0 )
            return error(engine_err_reader);
        if ( reader_gettoken( &r , type4 , 100 ) < 0 )
            return error(engine_err_reader);
            
        /* Check for wildcards. */
        wc[0] = ( strcmp( type1 , "X" ) == 0 );
        wc[1] = ( strcmp( type2 , "X" ) == 0 );
        wc[2] = ( strcmp( type3 , "X" ) == 0 );
        wc[3] = ( strcmp( type4 , "X" ) == 0 );
            
        /* Check if these types even exist. */
        if ( ( wc[0] || engine_gettype( e , type1 ) < 0 ) && 
             ( wc[1] || engine_gettype( e , type2 ) < 0 ) &&
             ( wc[2] || engine_gettype( e , type3 ) < 0 ) &&
             ( wc[3] || engine_gettype( e , type4 ) < 0 ) ) {
            if ( reader_skipline( &r ) < 0 )
                return error(engine_err_reader);
            continue;
            }
            
        /* Get the parameters K, n, and r0. */
        if ( reader_gettoken( &r , buff , 100 ) < 0 )
            return error(engine_err_reader);
        K = strtod( buff , &endptr );
        if ( *endptr != 0 )
            return error(engine_err_cpf);
        if ( reader_gettoken( &r , buff , 100 ) < 0 )
            return error(engine_err_reader);
        n = strtol( buff , &endptr , 0 );
        if ( *endptr != 0 )
            return error(engine_err_cpf);
        if ( reader_gettoken( &r , buff , 100 ) < 0 )
            return error(engine_err_reader);
        r0 = strtod( buff , &endptr );
        if ( *endptr != 0 )
            return error(engine_err_cpf);
            
        /* Run through the dihedral list and create the potential if necessary. */
        potid = -1;
        for ( k = 0 ; k < e->nr_dihedrals ; k++ ) {
        
            /* Does this dihedral match the types? */
            if ( ( e->dihedrals[k].pid == -1 ) &&
                 ( ( ( wc[0] || strcmp( e->types[e->s.partlist[e->dihedrals[k].i]->type].name , type1 ) == 0 ) &&
                     ( wc[1] || strcmp( e->types[e->s.partlist[e->dihedrals[k].j]->type].name , type2 ) == 0 ) &&
                     ( wc[2] || strcmp( e->types[e->s.partlist[e->dihedrals[k].k]->type].name , type3 ) == 0 ) &&
                     ( wc[3] || strcmp( e->types[e->s.partlist[e->dihedrals[k].l]->type].name , type4 ) == 0 ) ) ||
                   ( ( wc[3] || strcmp( e->types[e->s.partlist[e->dihedrals[k].i]->type].name , type4 ) == 0 ) &&
                     ( wc[2] || strcmp( e->types[e->s.partlist[e->dihedrals[k].j]->type].name , type3 ) == 0 ) &&
                     ( wc[1] || strcmp( e->types[e->s.partlist[e->dihedrals[k].k]->type].name , type2 ) == 0 ) &&
                     ( wc[0] || strcmp( e->types[e->s.partlist[e->dihedrals[k].l]->type].name , type1 ) == 0 ) ) ) ) {
                
                /* Do we need to create the potential? */
                if ( potid < 0 ) {
                    if ( ( p = potential_create_harmonic_dihedral( 4.184*K , n , M_PI/180*r0 , 1e-2 ) ) == NULL ) {
                        printf( "engine_read_cpf: failed to create dihedral with K=%e, n=%i, delta=%e.\n" , K , n , r0 );
                        return error(engine_err_potential);
                        }
                    if ( ( potid = engine_dihedral_addpot( e , p ) ) < 0 )
                        return error(engine_err);
                    /* printf( "engine_read_cpf: generated potential for dihedral %s %s %s %s in [%e,%e] with %i intervals.\n" ,
                        type1 , type2 , type3 , type4 , e->p_dihedral[potid]->a , e->p_dihedral[potid]->b , e->p_dihedral[potid]->n ); */
                    }
                
                /* Add the potential to the dihedral. */
                e->dihedrals[k].pid = potid;
                
                }
        
            }
            
        /* Skip the rest of this line. */
        if ( reader_skipline( &r ) < 0 )
            return error(engine_err_reader);
            
        } /* dihedral loop. */
        
    
    /* Skip the rest of the "IMPRoper" line. */
    if ( reader_skipline( &r ) < 0 )
        return error(engine_err_reader);
        
    /* Main improper dihedral loop. */
    while ( 1 ) {
    
        /* Get a token. */
        if ( reader_gettoken( &r , type1 , 100 ) < 0 )
            return error(engine_err_reader);
            
        /* If it's the start of the "NONBonded" section, break. */
        if ( strncmp( type1 , "NONB" , 4 ) == 0 )
            break;
            
        /* Otherwise, get the next three tokens, e.g. the second, third and fouth type. */
        if ( reader_gettoken( &r , type2 , 100 ) < 0 )
            return error(engine_err_reader);
        if ( reader_gettoken( &r , type3 , 100 ) < 0 )
            return error(engine_err_reader);
        if ( reader_gettoken( &r , type4 , 100 ) < 0 )
            return error(engine_err_reader);
            
        /* Check for wildcards. */
        wc[0] = ( strcmp( type1 , "X" ) == 0 );
        wc[1] = ( strcmp( type2 , "X" ) == 0 );
        wc[2] = ( strcmp( type3 , "X" ) == 0 );
        wc[3] = ( strcmp( type4 , "X" ) == 0 );
            
        /* Check if these types even exist. */
        if ( ( wc[0] || engine_gettype( e , type1 ) < 0 ) && 
             ( wc[1] || engine_gettype( e , type2 ) < 0 ) &&
             ( wc[2] || engine_gettype( e , type3 ) < 0 ) &&
             ( wc[3] || engine_gettype( e , type4 ) < 0 ) ) {
            if ( reader_skipline( &r ) < 0 )
                return error(engine_err_reader);
            continue;
            }
            
        /* Get the parameters K and r0. */
        if ( reader_gettoken( &r , buff , 100 ) < 0 )
            return error(engine_err_reader);
        K = strtod( buff , &endptr );
        if ( *endptr != 0 )
            return error(engine_err_cpf);
        if ( reader_skiptoken( &r ) < 0 )
            return error(engine_err_reader);
        if ( reader_gettoken( &r , buff , 100 ) < 0 )
            return error(engine_err_reader);
        r0 = strtod( buff , &endptr );
        if ( *endptr != 0 )
            return error(engine_err_cpf);
            
        /* Run through the dihedral list and create the potential if necessary. */
        potid = -1;
        for ( k = 0 ; k < e->nr_dihedrals ; k++ ) {
        
            /* Does this dihedral match the types? */
            if ( ( e->dihedrals[k].pid == -2 ) &&
                 ( ( ( wc[0] || strcmp( e->types[e->s.partlist[e->dihedrals[k].i]->type].name , type1 ) == 0 ) &&
                     ( wc[1] || strcmp( e->types[e->s.partlist[e->dihedrals[k].j]->type].name , type2 ) == 0 ) &&
                     ( wc[2] || strcmp( e->types[e->s.partlist[e->dihedrals[k].k]->type].name , type3 ) == 0 ) &&
                     ( wc[3] || strcmp( e->types[e->s.partlist[e->dihedrals[k].l]->type].name , type4 ) == 0 ) ) ||
                   ( ( wc[3] || strcmp( e->types[e->s.partlist[e->dihedrals[k].i]->type].name , type4 ) == 0 ) &&
                     ( wc[2] || strcmp( e->types[e->s.partlist[e->dihedrals[k].j]->type].name , type3 ) == 0 ) &&
                     ( wc[1] || strcmp( e->types[e->s.partlist[e->dihedrals[k].k]->type].name , type2 ) == 0 ) &&
                     ( wc[0] || strcmp( e->types[e->s.partlist[e->dihedrals[k].l]->type].name , type1 ) == 0 ) ) ) ) {
                
                /* Do we need to create the potential? */
                if ( potid < 0 ) {
                    if ( ( p = potential_create_harmonic_angle( M_PI/180*(r0-45) , M_PI/180*(r0+45) , 4.184*K , M_PI/180*r0 , tol ) ) == NULL )
                        return error(engine_err_potential);
                    if ( ( potid = engine_dihedral_addpot( e , p ) ) < 0 )
                        return error(engine_err);
                    /* printf( "engine_read_cpf: generated potential for imp. dihedral %s %s %s %s with %i intervals.\n" ,
                        type1 , type2 , type3 , type4 , e->p_dihedral[potid]->n ); */
                    }
                
                /* Add the potential to the dihedral. */
                e->dihedrals[k].pid = potid;
                
                }
        
            }
            
        /* Skip the rest of this line. */
        if ( reader_skipline( &r ) < 0 )
            return error(engine_err_reader);
            
        } /* dihedral loop. */
        
    
    /* Skip the rest of the "NONBonded" line. */
    if ( reader_skipline( &r ) < 0 )
        return error(engine_err_reader);
        
    /* Main loop for non-bonded interactions. */
    while ( 1 ) {
    
        /* Get the next token. */
        if ( reader_gettoken( &r , type1 , 100 ) < 0 )
            return error(engine_err_reader);
            
        /* Bail? */
        if ( strncmp( type1 , "HBOND" , 5 ) == 0 )
            break;
            
        /* Skip the first parameter. */
        if ( reader_skiptoken( &r ) < 0 )
            return error(engine_err_reader);
    
        /* Get the next two parameters. */
        if ( reader_gettoken( &r , buff , 100 ) < 0 )
            return error(engine_err_reader);
        K = strtod( buff , &endptr );
        if ( *endptr != 0 ) {
            printf( "engine_read_cpf: error parsing float token \"%s\" in non-bonded interactions.\n" , buff );
            return error(engine_err_cpf);
            }
        if ( reader_gettoken( &r , buff , 100 ) < 0 )
            return error(engine_err_reader);
        r0 = strtod( buff , &endptr );
        if ( *endptr != 0 )
            return error(engine_err_cpf);
            
        /* Run through the types and store the parameters for each match. */
        for ( k = 0 ; k < e->nr_types ; k++ )
            if ( strcmp( e->types[k].name , type1 ) == 0 ) {
                e->types[k].eps = 4.184 * K;
                e->types[k].rmin = 0.1 * r0;
                }
                
        /* Skip the rest of the line. */
        if ( reader_skipline( &r ) < 0 )
            return error(engine_err_reader);
            
        }
        
    /* Close the reader. */
    reader_close( &r );
        
    /* Loop over all the type pairs and construct the non-bonded potentials. */
    for ( j = 0 ; j < e->nr_types ; j++ )
        for ( k = j ; k < e->nr_types ; k++ ) {
            
            /* Has a potential been specified for this case? */
            if ( ( e->types[j].eps == 0.0 || e->types[k].eps == 0.0 ) &&
                 ( kappa < 0.0 || e->types[j].charge == 0.0 || e->types[k].charge == 0.0 ) )
                continue;
                
            /* Has this potential already been specified? */
            if ( kappa < 0.0 ) {
                for ( jj = 0 ; jj < j && ( e->types[jj].eps != e->types[j].eps || e->types[jj].rmin != e->types[j].rmin ) ; jj++ );
                for ( kk = 0 ; kk < k && ( e->types[kk].eps != e->types[k].eps || e->types[kk].rmin != e->types[k].rmin ) ; kk++ );
                }
            else {
                for ( jj = 0 ; jj < j && ( e->types[jj].eps != e->types[j].eps || e->types[jj].rmin != e->types[j].rmin || e->types[jj].charge != e->types[j].charge ) ; jj++ );
                for ( kk = 0 ; kk < k && ( e->types[kk].eps != e->types[k].eps || e->types[kk].rmin != e->types[k].rmin || e->types[kk].charge != e->types[k].charge ) ; kk++ );
                }
            if ( jj < j && kk < k ) {
                if ( e->p[ jj + e->max_type*kk ] != NULL && engine_addpot( e , e->p[ jj + e->max_type*kk ] , j , k ) < 0 )
                    return error(engine_err);
                continue;
                }
                    
            /* Construct the common LJ parameters. */
            K = sqrt( e->types[j].eps * e->types[k].eps );
            r0 = e->types[j].rmin + e->types[k].rmin;
            r2 = r0*r0; r6 = r2*r2*r2;
            A = K*r6*r6; B = 2*K*r6;
            q = e->types[j].charge*e->types[k].charge;
                
            /* Construct the potential. */
            /* printf( "engine_read_cpf: creating %s-%s potential with A=%e B=%e q=%e.\n" ,
                e->types[j].name , e->types[k].name , 
                K*r6*r6 , K*2*r6 , e->types[j].charge*e->types[k].charge ); */
            if ( K == 0.0 ) {
                if ( q != 0.0 && kappa >= 0.0 ) {
                    if ( kappa > 0.0 ) {
                        if ( ( p = potential_create_Ewald( 0.1 , e->s.cutoff , q , kappa , tol ) ) == NULL )
                            return error(engine_err_potential);
                        }
                    else {
                        if ( ( p = potential_create_Coulomb( 0.1 , e->s.cutoff , q , tol ) ) == NULL )
                            return error(engine_err_potential);
                        }
                    }
                else
                    p = NULL;
                }
            if ( kappa < 0.0 ) {
                al = r0/2;
                ar = r0;
                Kr0 = fabs( potential_LJ126( r0 , A , B ) );
                while ( ar-al > 1e-5 ) {
                    am = 0.5*(al + ar); vm = potential_LJ126( am , A , B );
                    if ( fabs(vm) < engine_maxKcutoff*Kr0 )
                        ar = am;
                    else
                        al = am;
                    }
                if ( ( p = potential_create_LJ126_switch( al , e->s.cutoff , A , B , 0.7 , tol ) ) == NULL ) {
                    printf( "engine_read_xplor: failed to create %s-%s potential with A=%e B=%e on [%e,%e].\n" ,
                    e->types[j].name , e->types[k].name , 
                    A , B , al , e->s.cutoff );
                    return error(engine_err_potential);
                    }
                }
            else if ( kappa == 0.0 ) {
                al = r0/2;
                ar = r0;
                Kr0 = fabs( potential_LJ126( r0 , A , B ) + potential_escale*q/r0 );
                while ( ar-al > 1e-5 ) {
                    am = 0.5*(al + ar); vm = potential_LJ126( am , A , B ) + potential_escale*q/am;
                    if ( fabs(vm) < engine_maxKcutoff*Kr0 )
                        ar = am;
                    else
                        al = am;
                    }
                if ( ( p = potential_create_LJ126_Coulomb( al , e->s.cutoff , A , B , q , tol ) ) == NULL ) {
                    printf( "engine_read_xplor: failed to create %s-%s potential with A=%e B=%e q=%e on [%e,%e].\n" ,
                    e->types[j].name , e->types[k].name , 
                    A ,B , q ,
                    al , e->s.cutoff );
                    return error(engine_err_potential);
                    }
                }
            else  {
                al = r0/2;
                ar = r0;
                Kr0 = fabs( potential_LJ126( r0 , A , B ) + q*potential_Ewald( r0, kappa ) );
                while ( ar-al > 1e-5 ) {
                    am = 0.5*(al + ar); vm = potential_LJ126( am , A , B ) + q*potential_Ewald( am , kappa );
                    if ( fabs(vm) < engine_maxKcutoff*Kr0 )
                        ar = am;
                    else
                        al = am;
                    }
                if ( ( p = potential_create_LJ126_Ewald_switch( al , e->s.cutoff , A , B , q , kappa , 0.7 , tol ) ) == NULL ) {
                    printf( "engine_read_xplor: failed to create %s-%s potential with A=%e B=%e q=%e on [%e,%e].\n" ,
                    e->types[j].name , e->types[k].name , 
                    A , B , q ,
                    al , e->s.cutoff );
                    return error(engine_err_potential);
                    }
                }
                
            /* Register it with the local authorities. */
            if ( p != NULL && engine_addpot( e , p , j , k ) < 0 )
                return error(engine_err);
                
            }
        
    /* It's been a hard day's night. */
    return engine_err_ok;
        
    }


/**
 * @brief Read the simulation setup from a PSF and PDB file pair.
 *
 * @param e The #engine.
 * @param psf The open PSF file.
 * @param pdb The open PDB file.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
 
int engine_read_psf ( struct engine *e , int psf , int pdb ) {

    struct reader r;
    char type[100], typename[100], buff[100], *endptr;
    int pid, pjd, pkd, pld, j, k, n, id, *resids, *typeids, typelen, bufflen;
    double q, m, x[3];
    struct particle p;
    
    /* Check inputs. */
    if ( e == NULL )
        return error(engine_err_null);
        
    /* Init the reader with the PSF file. */
    if ( reader_init( &r , psf , NULL , "!" , "\n" , engine_readbuff ) < 0 )
        return error(engine_err_reader);
        
    /* Read the PSF header token. */
    if ( reader_gettoken( &r , buff , 100 ) < 0 )
        return error(engine_err_reader);
    if ( strcmp( buff , "PSF" ) != 0 )
        return error(engine_err_psf);
    
    /* Ok, now read the number of comment lines and skip them. */
    if ( reader_gettoken( &r , buff , 100 ) < 0 )
        return error(engine_err_reader);
    n = strtol( buff , &endptr , 0 );
    if ( *endptr != 0 )
        return error(engine_err_psf);
    for ( k = 0 ; k <= n ; k++ )
        if ( reader_skipline( &r ) < 0 )
            return error(engine_err_reader);
            
    /* Now get the number of atoms, along with the comment. */
    if ( reader_gettoken( &r , buff , 100 ) < 0 )
        return error(engine_err_reader);
    n = strtol( buff , &endptr , 0 );
    if ( *endptr != 0 )
        return error(engine_err_psf);
    if ( reader_getcomment( &r , buff , 100 ) < 0 )
        return error(engine_err_reader);
    if ( strncmp( buff , "NATOM" , 5 ) != 0 )
        return error(engine_err_psf);
        
    /* Allocate memory for the type IDs. */
    if ( ( typeids = (int *)malloc( sizeof(int) * n ) ) == NULL ||
         ( resids = (int *)malloc( sizeof(int) * n ) ) == NULL )
        return error(engine_err_malloc);
        
    /* Loop over the atom list. */
    for ( k = 0 ; k < n ; k++ ) {
    
        /* Skip the first two tokens (ID, segment). */
        for ( j = 0 ; j < 2 ; j++ )
            if ( reader_skiptoken( &r ) < 0 )
                return error(engine_err_reader);
                
        /* Get the residue id. */
        if ( reader_gettoken( &r , buff , 100 ) < 0 )
            return error(engine_err_reader);
        resids[k] = strtol( buff , &endptr , 0 );
        if ( *endptr != 0 )
            return error(engine_err_psf);
        
        /* Skip the next two tokens (res name, atom name). */
        for ( j = 0 ; j < 2 ; j++ )
            if ( reader_skiptoken( &r ) < 0 )
                return error(engine_err_reader);
                
        /* Get the atom type. */
        if ( ( typelen = reader_gettoken( &r , type , 100 ) ) < 0 )
            return error(engine_err_reader);
    
        /* Get the atom charge. */
        if ( ( bufflen = reader_gettoken( &r , buff , 100 ) ) < 0 )
            return error(engine_err_reader);
        q = strtod( buff , &endptr );
        if ( *endptr != 0 )
            return error(engine_err_psf);
            
        /* Merge the type and charge. */
        memcpy( typename , type , typelen );
        memcpy( &typename[typelen] , buff , bufflen+1 );
        
        /* Get the atom mass. */
        if ( ( bufflen = reader_gettoken( &r , buff , 100 ) ) < 0 )
            return error(engine_err_reader);
        m = strtod( buff , &endptr );
        if ( *endptr != 0 )
            return error(engine_err_psf);

        /* Try to get a type id. */
        if ( ( id = engine_gettype2( e , typename ) ) >= 0 )
            typeids[k] = id;
        
        /* Otherwise, register a new type. */
        else if ( id == engine_err_range ) {
        
            if ( ( typeids[k] = engine_addtype( e , m , q , type , typename ) ) < 0 )
                return error(engine_err);
        
            }
            
        /* error? */
        else
            return error(engine_err);
            
        /* Read the trailing zero. */
        if ( reader_skiptoken( &r ) < 0 )
            return error(engine_err_reader);
    
        }
        
    /* Look for the number of bonds. */
    if ( reader_gettoken( &r , buff , 100 ) < 0 )
        return error(engine_err_reader);
    n = strtol( buff , &endptr , 0 );
    if ( *endptr != 0 )
        return error(engine_err_psf);
    if ( reader_getcomment( &r , buff , 100 ) < 0 )
        return error(engine_err_reader);
    if ( strncmp( buff , "NBOND" , 5 ) != 0 )
        return error(engine_err_psf);
        
    /* Load the bonds. */
    for ( k = 0 ; k < n ; k++ ) {
    
        /* Get the particle IDs. */
        if ( reader_gettoken( &r , buff , 100 ) < 0 )
            return error(engine_err_reader);
        pid = strtol( buff , &endptr , 0 );
        if ( *endptr != 0 )
            return error(engine_err_psf);
            
        if ( reader_gettoken( &r , buff , 100 ) < 0 )
            return error(engine_err_reader);
        pjd = strtol( buff , &endptr , 0 );
        if ( *endptr != 0 )
            return error(engine_err_psf);
            
        /* Store the bond. */
        if ( engine_bond_add( e , pid-1 , pjd-1 ) < 0 )
            return error(engine_err);
    
        }
                    
    /* Look for the number of angles. */
    if ( reader_gettoken( &r , buff , 100 ) < 0 )
        return error(engine_err_reader);
    n = strtol( buff , &endptr , 0 );
    if ( *endptr != 0 )
        return error(engine_err_psf);
    if ( reader_getcomment( &r , buff , 100 ) < 0 )
        return error(engine_err_reader);
    if ( strncmp( buff , "NTHETA" , 5 ) != 0 )
        return error(engine_err_psf);
        
    /* Load the angles. */
    for ( k = 0 ; k < n ; k++ ) {
    
        /* Get the particle IDs. */
        if ( reader_gettoken( &r , buff , 100 ) < 0 )
            return error(engine_err_reader);
        pid = strtol( buff , &endptr , 0 );
        if ( *endptr != 0 )
            return error(engine_err_psf);
            
        if ( reader_gettoken( &r , buff , 100 ) < 0 )
            return error(engine_err_reader);
        pjd = strtol( buff , &endptr , 0 );
        if ( *endptr != 0 )
            return error(engine_err_psf);
            
        if ( reader_gettoken( &r , buff , 100 ) < 0 )
            return error(engine_err_reader);
        pkd = strtol( buff , &endptr , 0 );
        if ( *endptr != 0 )
            return error(engine_err_psf);
            
        /* Store the angle, we will set the potential later. */
        if ( engine_angle_add( e , pid-1 , pjd-1 , pkd-1 , -1 ) < 0 )
            return error(engine_err);
    
        }
        
    /* Look for the number of dihedrals. */
    if ( reader_gettoken( &r , buff , 100 ) < 0 )
        return error(engine_err_reader);
    n = strtol( buff , &endptr , 0 );
    if ( *endptr != 0 )
        return error(engine_err_psf);
    if ( reader_getcomment( &r , buff , 100 ) < 0 )
        return error(engine_err_reader);
    if ( strncmp( buff , "NPHI" , 4 ) != 0 )
        return error(engine_err_psf);
        
    /* Load the dihedrals. */
    for ( k = 0 ; k < n ; k++ ) {
    
        /* Get the particle IDs. */
        if ( reader_gettoken( &r , buff , 100 ) < 0 )
            return error(engine_err_reader);
        pid = strtol( buff , &endptr , 0 );
        if ( *endptr != 0 )
            return error(engine_err_psf);
            
        if ( reader_gettoken( &r , buff , 100 ) < 0 )
            return error(engine_err_reader);
        pjd = strtol( buff , &endptr , 0 );
        if ( *endptr != 0 )
            return error(engine_err_psf);
            
        if ( reader_gettoken( &r , buff , 100 ) < 0 )
            return error(engine_err_reader);
        pkd = strtol( buff , &endptr , 0 );
        if ( *endptr != 0 )
            return error(engine_err_psf);
            
        if ( reader_gettoken( &r , buff , 100 ) < 0 )
            return error(engine_err_reader);
        pld = strtol( buff , &endptr , 0 );
        if ( *endptr != 0 )
            return error(engine_err_psf);
            
        /* Store the dihedral, we will set the potential later. */
        if ( engine_dihedral_add( e , pid-1 , pjd-1 , pkd-1 , pld-1 , -1 ) < 0 )
            return error(engine_err);
    
        }
        
    /* Look for the number of improper dihedrals. */
    if ( reader_gettoken( &r , buff , 100 ) < 0 )
        return error(engine_err_reader);
    n = strtol( buff , &endptr , 0 );
    if ( *endptr != 0 )
        return error(engine_err_psf);
    if ( reader_getcomment( &r , buff , 100 ) < 0 )
        return error(engine_err_reader);
    if ( strncmp( buff , "NIMP" , 4 ) != 0 )
        return error(engine_err_psf);
        
    /* Load the improper dihedrals. */
    for ( k = 0 ; k < n ; k++ ) {
    
        /* Get the particle IDs. */
        if ( reader_gettoken( &r , buff , 100 ) < 0 )
            return error(engine_err_reader);
        pid = strtol( buff , &endptr , 0 );
        if ( *endptr != 0 )
            return error(engine_err_psf);
            
        if ( reader_gettoken( &r , buff , 100 ) < 0 )
            return error(engine_err_reader);
        pjd = strtol( buff , &endptr , 0 );
        if ( *endptr != 0 )
            return error(engine_err_psf);
            
        if ( reader_gettoken( &r , buff , 100 ) < 0 )
            return error(engine_err_reader);
        pkd = strtol( buff , &endptr , 0 );
        if ( *endptr != 0 )
            return error(engine_err_psf);
            
        if ( reader_gettoken( &r , buff , 100 ) < 0 )
            return error(engine_err_reader);
        pld = strtol( buff , &endptr , 0 );
        if ( *endptr != 0 )
            return error(engine_err_psf);
            
        /* Store the dihedral, we will set the potential later. */
        if ( engine_dihedral_add( e , pid-1 , pjd-1 , pkd-1 , pld-1 , -2 ) < 0 )
            return error(engine_err);
    
        }
        
    /* There may be more stuff in the file, but we'll ignore that for now! */
    reader_close( &r );
    
    /* Init the reader with the PDb file. */
    if ( reader_init( &r , pdb , NULL , NULL , NULL , engine_readbuff ) < 0 )
        return error(engine_err_reader);
        
    /* Init the part data. */
    bzero( &p , sizeof(struct particle) );
    pid = 0;
        
    /* Main loop. */
    while ( 1 ) {
    
        /* Get a token. */
        if ( reader_gettoken( &r , buff , 100 ) < 0 )
            return error(engine_err_reader);
            
        /* If it's a REMARK, just skip the line. */
        if ( strncmp( buff , "REMARK" , 6 ) == 0 ) {
            if ( reader_skipline( &r ) < 0 )
                return error(engine_err_reader);
            }
            
        /* Is it an atom? */
        else if ( strncmp( buff , "ATOM" , 4 ) == 0 ) {
        
            /* Get the atom ID. */
            /* if ( reader_gettoken( &r , buff , 100 ) < 0 )
                return error(engine_err_reader);
            pid = strtol( buff , &endptr , 0 );
            if ( *endptr != 0 )
                return error(engine_err_pdb); */
            if ( reader_skiptoken( &r ) < 0 )
                return error(engine_err_reader);
            pid += 1;
                
            /* Get the atom type. */
            if ( ( typelen = reader_gettoken( &r , type , 100 ) ) < 0 )
                return error(engine_err_reader);
                
            /* Does the type match the PSF data? */
            /* if ( strncmp( e->types[typeids[pid-1]].name , type , typelen ) != 0 )
                return error(engine_err_pdb); */
                
            /* Ignore the two following tokens. */
            if ( reader_skiptoken( &r ) < 0 )
                return error(engine_err_reader);
            if ( reader_skiptoken( &r ) < 0 )
                return error(engine_err_reader);
                
            /* Load the position. */
            for ( k = 0 ; k < 3 ; k++ ) {
                if ( ( bufflen = reader_gettoken( &r , buff , 100 ) ) < 0 )
                    return error(engine_err_reader);
                x[k] = fmod( e->s.dim[k] - e->s.origin[k] + 0.1 * strtod( buff , &endptr ) , e->s.dim[k] ) + e->s.origin[k];
                if ( *endptr != 0 ) {
                    printf( "engine_read_psf: error reading %ith entry (%s), got position \"%s\".\n" , pid , type , buff );
                    return error(engine_err_pdb);
                    }
                }
                
            /* Add a part of the given type at the given location. */
            p.id = pid-1;
            p.vid = resids[pid-1];
            p.q = e->types[typeids[pid-1]].charge;
            p.flags = PARTICLE_FLAG_NONE;
            p.type = typeids[pid-1];
            if ( space_addpart( &e->s , &p , x ) < 0 )
                return error(engine_err_space);
                
            /* Skip the rest of the line. */
            if ( reader_skipline( &r ) < 0 )
                return error(engine_err_reader);
                
            }
            
        /* Is it the end? */
        else if ( strncmp( buff , "END" , 3 ) == 0 )
            break;
            
        /* Otherwise, it's and error. */
        else
            return error(engine_err_pdb);
            
            
        } /* main PDB loop. */
        
    /* Clean up allocs. */
    reader_close( &r );
    free(typeids);
    free(resids);
                    
    /* We're on the road again! */
    return engine_err_ok;

    }
    

/**
 * @brief Dump the contents of the enginge to a PSF and PDB file.
 *
 * @param e The #engine.
 * @param psf A pointer to @c FILE to which to write the PSF file.
 * @param pdb A pointer to @c FILE to which to write the PDB file.
 *
 * If any of @c psf or @c pdb are @c NULL, the respective output will
 * not be generated.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
 
int engine_dump_PSF ( struct engine *e , FILE *psf , FILE *pdb , char *excl[] , int nr_excl ) {

    struct space *s;
    struct space_cell *c;
    struct particle *p;
    int k, pid, bid, aid;

    /* Check inputs. */
    if ( e == NULL || ( psf == NULL && pdb == NULL ) )
        return error(engine_err_null);
        
    /* Get a hold of the space. */
    s = &e->s;
        
    /* Write the header for the psf file if needed. */
    if ( psf != NULL )
        fprintf( psf , "PSF\n0 !NTITLE\n%i !NATOM\n" , s->nr_parts );
        
    /* Loop over the cells and parts. */
    for ( pid = 0 ; pid < s->nr_parts ; pid++ ) {
        if ( ( p = s->partlist[pid] ) == NULL || ( c = s->celllist[pid] ) == NULL )
            continue;
        for ( k = 0 ; k < nr_excl ; k++ )
            if ( strcmp( e->types[p->type].name , excl[k] ) == 0 )
                break;
        if ( nr_excl > 0 && k < nr_excl )
            continue;
        if ( pdb != NULL )
            fprintf( pdb , "ATOM  %5d %4s %3s X%4i    %8.3f%8.3f%8.3f\n" ,
                (p->id+1)%100000 , e->types[p->type].name , "" , (p->vid+1)%10000 ,
                10 * ( p->x[0] + c->origin[0] ) , 10 * ( p->x[1] + c->origin[1] ) , 10 * ( p->x[2] + c->origin[2] ) );
        if ( psf != NULL )
            fprintf( psf , "%8i %4s %4i %4s %4s %4s %15.6f %15.6f    0\n" ,
                p->id+1 , "WAT" , p->vid+1 , "TIP3" , e->types[p->type].name , e->types[p->type].name , e->types[p->type].charge , e->types[p->type].mass );
        }
        
    /* Close-up the PDB file. */
    if ( pdb != NULL )
        fprintf( pdb , "END\n" );
        
    /* Dump bonds and angles to PSF? */
    if ( psf != NULL ) {
    
        /* Dump bonds. */
        fprintf( psf , "%i !NBOND\n" , e->nr_bonds + e->nr_angles );
        for ( bid = 0 ; bid < e->nr_bonds ; bid++ )
            if ( bid % 4 == 3 )
                fprintf( psf , " %i %i\n" , e->bonds[bid].i+1 , e->bonds[bid].j+1 );
            else
                fprintf( psf , " %i %i" , e->bonds[bid].i+1 , e->bonds[bid].j+1 );
        for ( aid = 0 ; aid < e->nr_angles ; aid++ )
            if ( aid % 4 == 3 )
                fprintf( psf , " %i %i\n" , e->angles[aid].i+1 , e->angles[aid].k+1 );
            else
                fprintf( psf , " %i %i" , e->angles[aid].i+1 , e->angles[aid].k+1 );
                
        /* Dump angles. */
        fprintf( psf , "%i !NTHETA\n" , e->nr_angles );
        for ( aid = 0 ; aid < e->nr_angles ; aid++ )
            if ( aid % 3 == 2 )
                fprintf( psf , " %i %i %i\n" , e->angles[aid].i+1 , e->angles[aid].j+1 , e->angles[aid].k+1 );
            else
                fprintf( psf , " %i %i %i" , e->angles[aid].i+1 , e->angles[aid].j+1 , e->angles[aid].k+1 );
                
        /* Dump remaining bogus headers. */
        fprintf( psf , "0 !NPHI\n" );
        fprintf( psf , "0 !NIMPHI\n" );
        fprintf( psf , "0 !NDON\n" );
        fprintf( psf , "0 !NACC\n" );
        fprintf( psf , "0 !NNB\n" );
        
        }
        
    /* We're on a road to nowhere... */
    return engine_err_ok;

    }


