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
#ifndef INCLUDE_POTENTIAL_H_
#define INCLUDE_POTENTIAL_H_
#include "platform.h"

MDCORE_BEGIN_DECLS

/* potential error codes */
#define potential_err_ok                    0
#define potential_err_null                  -1
#define potential_err_malloc                -2
#define potential_err_bounds                -3
#define potential_err_nyi                   -4
#define potential_err_ivalsmax              -5


/* some constants */
#define potential_degree                    5
#define potential_chunk                     (potential_degree+3)
#define potential_ivalsa                    1
#define potential_ivalsb                    10
#define potential_N                         100
#define potential_align                     64
#define potential_align                     64
#define potential_ivalsmax                  320

#define potential_escale                    (0.079577471545947667882)
// #define potential_escale                    1.0


/* potential flags */
#define potential_flag_none                  0
#define potential_flag_LJ126                 1
#define potential_flag_Ewald                 2
#define potential_flag_Coulomb               4
#define potential_flag_single                6


/** ID of the last error. */
extern int potential_err;


/** The #potential structure. */
typedef struct potential {

	/** Coefficients for the interval transform. */
	FPTYPE alpha[4];

	/** The coefficients. */
	FPTYPE *c;

	/** Interval edges. */
	double a, b;

	/** Flags. */
	unsigned int flags;

	/** Nr of intervals. */
	int n;

} potential;


/** Fictitious null potential. */
extern struct potential potential_null;


/* associated functions */
void potential_clear ( struct potential *p );
int potential_init ( struct potential *p , double (*f)( double ) , double (*fp)( double ) , double (*f6p)( double ) , FPTYPE a , FPTYPE b , FPTYPE tol );
int potential_getcoeffs ( double (*f)( double ) , double (*fp)( double ) , FPTYPE *xi , int n , FPTYPE *c , FPTYPE *err );
double potential_getalpha ( double (*f6p)( double ) , double a , double b );
struct potential *potential_create_LJ126 ( double a , double b , double A , double B , double tol );
struct potential *potential_create_LJ126_switch ( double a , double b , double A , double B , double s , double tol );
struct potential *potential_create_LJ126_Ewald ( double a , double b , double A , double B , double q , double kappa , double tol );
struct potential *potential_create_LJ126_Ewald_switch ( double a , double b , double A , double B , double q , double kappa , double s , double tol );
struct potential *potential_create_LJ126_Coulomb ( double a , double b , double A , double B , double q , double tol );
struct potential *potential_create_Ewald ( double a , double b , double q , double kappa , double tol );
struct potential *potential_create_Coulomb ( double a , double b , double q , double tol );
struct potential *potential_create_harmonic ( double a , double b , double K , double r0 , double tol );
struct potential *potential_create_harmonic_angle ( double a , double b , double K , double theta0 , double tol );
struct potential *potential_create_harmonic_dihedral ( double K , int n , double delta , double tol );

/* These functions are now all in potential_eval.h. */
/*
    void potential_eval ( struct potential *p , FPTYPE r2 , FPTYPE *e , FPTYPE *f );
    void potential_eval_expl ( struct potential *p , FPTYPE r2 , FPTYPE *e , FPTYPE *f );
    void potential_eval_vec_4single ( struct potential *p[4] , float *r2 , float *e , float *f );
    void potential_eval_vec_4single_r ( struct potential *p[4] , float *r_in , float *e , float *f );
    void potential_eval_vec_8single ( struct potential *p[4] , float *r2 , float *e , float *f );
    void potential_eval_vec_2double ( struct potential *p[4] , FPTYPE *r2 , FPTYPE *e , FPTYPE *f );
    void potential_eval_vec_4double ( struct potential *p[4] , FPTYPE *r2 , FPTYPE *e , FPTYPE *f );
    void potential_eval_vec_4double_r ( struct potential *p[4] , FPTYPE *r , FPTYPE *e , FPTYPE *f );
    void potential_eval_r ( struct potential *p , FPTYPE r , FPTYPE *e , FPTYPE *f );
 */

/* helper functions */
double potential_LJ126 ( double r , double A , double B );
double potential_LJ126_p ( double r , double A , double B );
double potential_LJ126_6p ( double r , double A , double B );
double potential_Ewald ( double r , double kappa );
double potential_Ewald_p ( double r , double kappa );
double potential_Ewald_6p ( double r , double kappa );
double potential_Coulomb ( double r );
double potential_Coulomb_p ( double r );
double potential_Coulomb_6p ( double r );
double potential_switch ( double r , double A , double B );
double potential_switch_p ( double r , double A , double B );

MDCORE_END_DECLS
#endif // INCLUDE_POTENTIAL_H_
