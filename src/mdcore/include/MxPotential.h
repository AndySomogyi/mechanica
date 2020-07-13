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
#include "fptype.h"
#include "carbon.h"

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
#define potential_ivalsmax                  640

#define potential_escale                    (0.079577471545947667882)
// #define potential_escale                    1.0


/* potential flags */
#define potential_flag_none                  0
#define potential_flag_LJ126                 1
#define potential_flag_Ewald                 2
#define potential_flag_Coulomb               4
#define potential_flag_single                6

/** flag defined for r^2 input */
#define potential_flag_r2                    1 << 0

/** potential defined for r input (no sqrt) */
#define potential_flag_r                     1 << 1

/** potential has Coulomb component */
#define potential_flag_coulomb               1 << 2

/** potential has LJ component */
#define potential_flag_lennard_jones         1 << 3

/** potential defined for angle */
#define potential_flag_angle                 1 << 4

/** potential defined for harmonic */
#define potential_flag_harmonic              1 << 5

/** potential defined for harmonic */
#define potential_flag_dihedral              1 << 6

/** potential defined for harmonic */
#define potential_flag_ewald                 1 << 7

/** potential defined for switch */
#define potential_flag_switch                1 << 8


/** ID of the last error. */
CAPI_DATA(int) potential_err;

typedef void (*MxPotentialEval) ( struct MxPotential *p , struct MxParticle *,
    struct MxParticle *b, FPTYPE r2 , FPTYPE *e , FPTYPE *f );

typedef struct MxPotential* (*MxPotentialCreate) (
    struct MxPotential *partial_potential,
    struct MxParticleType *a, struct MxParticleType *b );


/** The #potential structure. */
typedef struct MxPotential : PyObject {
    MxPotentialEval eval;

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
    
    MxPotentialCreate create_func;

} MxPotential;


/** Fictitious null potential. */
CAPI_DATA(struct MxPotential) potential_null;


/* associated functions */
CAPI_FUNC(void) potential_clear ( struct MxPotential *p );
CAPI_FUNC(int) potential_init ( struct MxPotential *p , double (*f)( double ) ,
							   double (*fp)( double ) , double (*f6p)( double ) ,
							   FPTYPE a , FPTYPE b , FPTYPE tol );

CAPI_FUNC(int) potential_getcoeffs ( double (*f)( double ) , double (*fp)( double ) ,
									 FPTYPE *xi , int n , FPTYPE *c , FPTYPE *err );

CAPI_FUNC(double) potential_getalpha ( double (*f6p)( double ) , double a , double b );

CAPI_FUNC(struct MxPotential *) potential_create_LJ126 ( double a , double b ,
														 double A , double B , double tol );
CAPI_FUNC(struct MxPotential *) potential_create_LJ126_switch ( double a , double b ,
																double A , double B ,
																double s , double tol );
CAPI_FUNC(struct MxPotential *) potential_create_LJ126_Ewald ( double a , double b ,
															   double A , double B ,
															   double q , double kappa ,
															   double tol );
CAPI_FUNC(struct MxPotential *) potential_create_LJ126_Ewald_switch ( double a , double b ,
																	  double A , double B ,
																	  double q , double kappa ,
																	  double s , double tol );

CAPI_FUNC(struct MxPotential *) potential_create_LJ126_Coulomb ( double a , double b ,
																 double A , double B ,
																 double q , double tol );
CAPI_FUNC(struct MxPotential *) potential_create_Ewald ( double a , double b ,
														 double q , double kappa ,
														 double tol );
CAPI_FUNC(struct MxPotential *) potential_create_Coulomb ( double a , double b ,
														   double q , double tol );
CAPI_FUNC(struct MxPotential *) potential_create_harmonic ( double a , double b ,
															double K , double r0 ,
															double tol );
CAPI_FUNC(struct MxPotential *) potential_create_harmonic_angle ( double a , double b ,
																  double K , double theta0 ,
																  double tol );
CAPI_FUNC(struct MxPotential *) potential_create_harmonic_dihedral ( double K , int n ,
																	 double delta , double tol );


CAPI_FUNC(struct MxPotential *) potential_create_SS1(double k, double e, double r0, double a , double b ,double tol);

CAPI_FUNC(struct MxPotential *) potential_create_SS(int eta, double k, double e, double r0, double a , double b , double tol);

CAPI_FUNC(struct MxPotential *) potential_create_SS2(double k, double e, double r0, double a , double b ,double tol);


/**
 * @brief partially Creates a #potential representing a 12-6 Lennard-Jones potential
 *
 * squirrely design, but we want to partially create one of these based on just
 * sigma, the interaction strength, and fully create them when we bind them to
 * particles. We do this to keep the 'bind' idea consistent for the public API.
 *
 * the partial created potential has the epsilon, min max params, but we hold
 * off with the particle radii untill we bind them.
 */
CAPI_FUNC(struct MxPotential *) potential_partial_create_particle_radius (
    double sigma, double min_rad, double max_rad, double tol );


/**
 * @brief partially Creates a #potential representing a 12-6 Lennard-Jones potential
 *
 * squirrely design, but we want to partially create one of these based on just
 * sigma, the interaction strength, and fully create them when we bind them to
 * particles. We do this to keep the 'bind' idea consistent for the public API.
 *
 * @param a particle type a
 * @param b particle type b
 * @param sigma: strength of the interaction
 * @param min_rad The smallest radius for which the potential will be constructed.
 * @param max_rad The largest radius for which the potential will be constructed.
 * @param tol The tolerance to which the interpolation should match the exact
 *      potential.
 *
 * @return A newly-allocated #potential representing the potential
 *      @f$ \left( \frac{A}{r^{12}} - \frac{B}{r^6} \right) @f$ in @f$[a,b]@f$
 *      or @c NULL on error (see #potential_err).
 */
CAPI_FUNC(struct MxPotential *) potential_create_particle_radius (
    struct MxPotential *partial_potential,
    struct MxParticleType *a, struct MxParticleType *b );



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
CAPI_FUNC(double) potential_LJ126 ( double r , double A , double B );
CAPI_FUNC(double) potential_LJ126_p ( double r , double A , double B );
CAPI_FUNC(double) potential_LJ126_6p ( double r , double A , double B );
CAPI_FUNC(double) potential_Ewald ( double r , double kappa );
CAPI_FUNC(double) potential_Ewald_p ( double r , double kappa );
CAPI_FUNC(double) potential_Ewald_6p ( double r , double kappa );
CAPI_FUNC(double) potential_Coulomb ( double r );
CAPI_FUNC(double) potential_Coulomb_p ( double r );
CAPI_FUNC(double) potential_Coulomb_6p ( double r );
CAPI_FUNC(double) potential_switch ( double r , double A , double B );
CAPI_FUNC(double) potential_switch_p ( double r , double A , double B );



/**
 * The type of each individual particle.
 */
CAPI_DATA(PyTypeObject) MxPotential_Type;

HRESULT MxPotential_init(PyObject *m);

MDCORE_END_DECLS
#endif // INCLUDE_POTENTIAL_H_
