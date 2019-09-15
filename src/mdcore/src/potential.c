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
#include <math.h>
#include <float.h>
#include <string.h>

/* include local headers */
#include "errs.h"
#include "fptype.h"
#include "potential.h"
#include "potential_eval.h"

/** Macro to easily define vector types. */
#define simd_vector(elcount, type)  __attribute__((vector_size((elcount)*sizeof(type)))) type

/** The last error */
int potential_err = potential_err_ok;


/** The null potential */
FPTYPE c_null[] = { FPTYPE_ZERO , FPTYPE_ZERO , FPTYPE_ZERO , FPTYPE_ZERO , FPTYPE_ZERO , FPTYPE_ZERO , FPTYPE_ZERO , FPTYPE_ZERO };
struct potential potential_null = { { FPTYPE_ZERO , FPTYPE_ZERO , FPTYPE_ZERO , FPTYPE_ZERO } , c_null , 0.0 , DBL_MAX , potential_flag_none , 1 };


/* the error macro. */
#define error(id)				( potential_err = errs_register( id , potential_err_msg[-(id)] , __LINE__ , __FUNCTION__ , __FILE__ ) )

/* list of error messages. */
char *potential_err_msg[6] = {
		"Nothing bad happened.",
		"An unexpected NULL pointer was encountered.",
		"A call to malloc failed, probably due to insufficient memory.",
		"The requested value was out of bounds.",
		"Not yet implemented.",
		"Maximum number of intervals reached before tolerance satisfied."
};


/**
 * @brief Switching function.
 *
 * @param r The radius.
 * @param A The start of the switching region.
 * @param B The end of the switching region.
 */

inline double potential_switch ( double r , double A , double B ) {

	if ( r < A )
		return 1.0;
	else if ( r > B )
		return 0.0;
	else {

		double B2 = B*B, A2 = A*A, r2 = r*r;
		double B2mr2 = B2 - r2, B2mA2 = B2 - A2;

		return B2mr2*B2mr2 * ( B2 + 2*r2 - 3*A2 ) / ( B2mA2 * B2mA2 * B2mA2 );

	}

}

inline double potential_switch_p ( double r , double A , double B ) {

	if ( A < r && r < B ) {

		double B2 = B*B, A2 = A*A, r2 = r*r;
		double B2mr2 = B2 - r2, B2mA2 = B2 - A2;
		double r2_p = 2*r, B2mr2_p = -r2_p;

		return ( 2*B2mr2_p*B2mr2 * ( B2 + 2*r2 - 3*A2 ) + B2mr2*B2mr2 * 2*r2_p ) / ( B2mA2 * B2mA2 * B2mA2 );

	}
	else
		return 0.0;

}


/**
 * @brief A basic 12-6 Lennard-Jones potential.
 *
 * @param r The interaction radius.
 * @param A First parameter of the potential.
 * @param B Second parameter of the potential.
 *
 * @return The potential @f$ \left( \frac{A}{r^{12}} - \frac{B}{r^6} \right) @f$
 *      evaluated at @c r.
 */

inline double potential_LJ126 ( double r , double A , double B ) {

	double ir = 1.0/r, ir2 = ir * ir, ir6 = ir2*ir2*ir2, ir12 = ir6 * ir6;

	return ( A * ir12 - B * ir6 );

}

/**
 * @brief A basic 12-6 Lennard-Jones potential (first derivative).
 *
 * @param r The interaction radius.
 * @param A First parameter of the potential.
 * @param B Second parameter of the potential.
 *
 * @return The first derivative of the potential
 *      @f$ \left( \frac{A}{r^{12}} - \frac{B}{r^6} \right) @f$
 *      evaluated at @c r.
 */

inline double potential_LJ126_p ( double r , double A , double B ) {

	double ir = 1.0/r, ir2 = ir*ir, ir4 = ir2*ir2, ir12 = ir4*ir4*ir4;

	return 6.0 * ir * ( -2.0 * A * ir12 + B * ir4 * ir2 );

}

/**
 * @brief A basic 12-6 Lennard-Jones potential (sixth derivative).
 *
 * @param r The interaction radius.
 * @param A First parameter of the potential.
 * @param B Second parameter of the potential.
 *
 * @return The sixth derivative of the potential
 *      @f$ \left( \frac{A}{r^{12}} - \frac{B}{r^6} \right) @f$
 *      evaluated at @c r.
 */

inline double potential_LJ126_6p ( double r , double A , double B ) {

	double r2 = r * r, ir2 = 1.0 / r2, ir6 = ir2*ir2*ir2, ir12 = ir6 * ir6;

	return 10080.0 * ir12 * ( 884.0 * A * ir6 - 33.0 * B );

}

/**
 * @brief The Coulomb potential.
 *
 * @param r The interaction radius.
 *
 * @return The potential @f$ \frac{1}{4\pi r} @f$
 *      evaluated at @c r.
 */

inline double potential_Coulomb ( double r ) {

	return potential_escale / r;

}

/**
 * @brief The Coulomb potential (first derivative).
 *
 * @param r The interaction radius.
 *
 * @return The first derivative of the potential @f$ \frac{1}{4\pi r} @f$
 *      evaluated at @c r.
 */

inline double potential_Coulomb_p ( double r ) {

	return -potential_escale / (r*r);

}

/**
 * @brief TheCoulomb potential (sixth derivative).
 *
 * @param r The interaction radius.
 *
 * @return The sixth derivative of the potential @f$ \frac{1}{4\pi r} @f$
 *      evaluated at @c r.
 */

inline double potential_Coulomb_6p ( double r ) {

	double r2 = r*r, r4 = r2*r2, r7 = r*r2*r4;

	return 720.0 * potential_escale / r7;

}


/**
 * @brief The short-range part of an Ewald summation.
 *
 * @param r The interaction radius.
 * @param kappa The screening length of the Ewald summation.
 *
 * @return The potential @f$ \frac{\mbox{erfc}( \kappa r )}{r} @f$
 *      evaluated at @c r.
 */

inline double potential_Ewald ( double r , double kappa ) {

	return potential_escale * erfc( kappa * r ) / r;

}

/**
 * @brief The short-range part of an Ewald summation (first derivative).
 *
 * @param r The interaction radius.
 * @param kappa The screening length of the Ewald summation.
 *
 * @return The first derivative of the potential @f$ \frac{\mbox{erfc}( \kappa r )}{r} @f$
 *      evaluated at @c r.
 */

inline double potential_Ewald_p ( double r , double kappa ) {

	double r2 = r*r, ir = 1.0 / r, ir2 = ir*ir;
	const double isqrtpi = 0.56418958354775628695;

	return potential_escale * ( -2.0 * exp( -kappa*kappa * r2 ) * kappa * ir * isqrtpi -
			erfc( kappa * r ) * ir2 );

}

/**
 * @brief The short-range part of an Ewald summation (sixth derivative).
 *
 * @param r The interaction radius.
 * @param kappa The screening length of the Ewald summation.
 *
 * @return The sixth derivative of the potential @f$ \frac{\mbox{erfc}( \kappa r )}{r} @f$
 *      evaluated at @c r.
 */

inline double potential_Ewald_6p ( double r , double kappa ) {

	double r2 = r*r, ir2 = 1.0 / r2, r4 = r2*r2, ir4 = ir2*ir2, ir6 = ir2*ir4;
	double kappa2 = kappa*kappa;
	double t6, t23;
	const double isqrtpi = 0.56418958354775628695;

	t6 = erfc(kappa*r);
	t23 = exp(-kappa2*r2);
	return potential_escale * ( 720.0*t6/r*ir6+(1440.0*ir6+(960.0*ir4+(384.0*ir2+(144.0+(-128.0*r2+64.0*kappa2*r4)*kappa2)*kappa2)*kappa2)*kappa2)*kappa*isqrtpi*t23 );

}


double potential_create_harmonic_K;
double potential_create_harmonic_r0;

/* the potential functions */
double potential_create_harmonic_f ( double r ) {
	return potential_create_harmonic_K * ( r - potential_create_harmonic_r0 ) * ( r - potential_create_harmonic_r0 );
}

double potential_create_harmonic_dfdr ( double r ) {
	return 2.0 * potential_create_harmonic_K * ( r - potential_create_harmonic_r0 );
}

double potential_create_harmonic_d6fdr6 ( double r ) {
	return 0;
}

/**
 * @brief Creates a harmonic bond #potential
 *
 * @param a The smallest radius for which the potential will be constructed.
 * @param b The largest radius for which the potential will be constructed.
 * @param K The energy of the bond.
 * @param r0 The minimum energy distance.
 * @param tol The tolerance to which the interpolation should match the exact
 *      potential.
 *
 * @return A newly-allocated #potential representing the potential
 *      @f$ K(r-r_0)^2 @f$ in @f$[a,b]@f$
 *      or @c NULL on error (see #potential_err).
 */

struct potential *potential_create_harmonic ( double a , double b , double K , double r0 , double tol ) {

	struct potential *p;

	/* allocate the potential */
	if ( posix_memalign( (void **)&p , 16 , sizeof( struct potential ) ) != 0 ) {
		error(potential_err_malloc);
		return NULL;
	}

	/* fill this potential */
	potential_create_harmonic_K = K;
	potential_create_harmonic_r0 = r0;
	if ( potential_init( p , &potential_create_harmonic_f , NULL , &potential_create_harmonic_d6fdr6 , a , b , tol ) < 0 ) {
		free(p);
		return NULL;
	}

	/* return it */
			return p;

}


double potential_create_harmonic_dihedral_K;
int potential_create_harmonic_dihedral_n;
double potential_create_harmonic_dihedral_delta;

/* the potential functions */
double potential_create_harmonic_dihedral_f ( double r ) {
	double T[potential_create_harmonic_dihedral_n+1], U[potential_create_harmonic_dihedral_n+1];
	double cosd = cos(potential_create_harmonic_dihedral_delta), sind = sin(potential_create_harmonic_dihedral_delta);
	int k;
	T[0] = 1.0; T[1] = r;
	U[0] = 1.0; U[1] = 2*r;
	for ( k = 2 ; k <= potential_create_harmonic_dihedral_n ; k++ ) {
		T[k] = 2 * r * T[k-1] - T[k-2];
		U[k] = 2 * r * U[k-1] - U[k-2];
	}
	if ( potential_create_harmonic_dihedral_delta == 0.0 )
		return potential_create_harmonic_dihedral_K * ( 1.0 + T[potential_create_harmonic_dihedral_n] );
	else if ( potential_create_harmonic_dihedral_delta == M_PI )
		return potential_create_harmonic_dihedral_K * ( 1.0 - T[potential_create_harmonic_dihedral_n] );
	else if ( fabs(r) < 1.0 )
		return potential_create_harmonic_dihedral_K * ( 1.0 + T[potential_create_harmonic_dihedral_n]*cosd + U[potential_create_harmonic_dihedral_n-1]*sqrt(1.0-r*r)*sind );
	else
		return potential_create_harmonic_dihedral_K * ( 1.0 + T[potential_create_harmonic_dihedral_n]*cosd );
}

double potential_create_harmonic_dihedral_dfdr ( double r ) {
	double T[potential_create_harmonic_dihedral_n+1], U[potential_create_harmonic_dihedral_n+1];
	double cosd = cos(potential_create_harmonic_dihedral_delta), sind = sin(potential_create_harmonic_dihedral_delta);
	int k;
	T[0] = 1.0; T[1] = r;
	U[0] = 1.0; U[1] = 2*r;
	for ( k = 2 ; k <= potential_create_harmonic_dihedral_n ; k++ ) {
		T[k] = 2 * r * T[k-1] - T[k-2];
		U[k] = 2 * r * U[k-1] - U[k-2];
	}
	if ( potential_create_harmonic_dihedral_delta == 0.0 )
		return potential_create_harmonic_dihedral_K * potential_create_harmonic_dihedral_n*U[potential_create_harmonic_dihedral_n-1];
	else if ( potential_create_harmonic_dihedral_delta == M_PI )
		return -potential_create_harmonic_dihedral_K * potential_create_harmonic_dihedral_n*U[potential_create_harmonic_dihedral_n-1];
	else
		return potential_create_harmonic_dihedral_K * ( potential_create_harmonic_dihedral_n*U[potential_create_harmonic_dihedral_n-1]*cosd + ( 2*r*U[potential_create_harmonic_dihedral_n-1] - potential_create_harmonic_dihedral_n*T[potential_create_harmonic_dihedral_n] ) * sind / sqrt(1.0 - r*r) );
}

double potential_create_harmonic_dihedral_d6fdr6 ( double r ) {
	return 0.0;
}

/**
 * @brief Creates a harmonic dihedral #potential
 *
 * @param K The energy of the dihedral.
 * @param n The multiplicity of the dihedral.
 * @param delta The minimum energy dihedral.
 * @param tol The tolerance to which the interpolation should match the exact
 *      potential.
 *
 * @return A newly-allocated #potential representing the potential
 *      @f$ K(1 + \cos(n\arccos(r)-delta) @f$ in @f$[-1,1]@f$
 *      or @c NULL on error (see #potential_err).
 */

struct potential *potential_create_harmonic_dihedral ( double K , int n , double delta , double tol ) {

	struct potential *p;
	double a = -1.0, b = 1.0;

	/* Adjust end-points if delta is not a multiple of pi. */
	if ( fmod( delta , M_PI ) != 0 ) {
		a = -1.0 / (1.0 + sqrt(FPTYPE_EPSILON));
		b = 1.0 / (1.0 + sqrt(FPTYPE_EPSILON));
	}

	/* allocate the potential */
	if ( posix_memalign( (void **)&p , 16 , sizeof( struct potential ) ) != 0 ) {
		error(potential_err_malloc);
		return NULL;
	}

	/* fill this potential */
	potential_create_harmonic_dihedral_K = K;
	potential_create_harmonic_dihedral_n = n;
	potential_create_harmonic_dihedral_delta = delta;
	if ( potential_init( p , &potential_create_harmonic_dihedral_f , NULL , &potential_create_harmonic_dihedral_d6fdr6 , a , b , tol ) < 0 ) {
		free(p);
		return NULL;
	}

	/* return it */
			return p;

}


double potential_create_harmonic_angle_K;
double potential_create_harmonic_angle_theta0;

/* the potential functions */
double potential_create_harmonic_angle_f ( double r ) {
	double theta;
	r = fmin( 1.0 , fmax( -1.0 , r ) );
	theta = acos( r );
	return potential_create_harmonic_angle_K * ( theta - potential_create_harmonic_angle_theta0 ) * ( theta - potential_create_harmonic_angle_theta0 );
}

double potential_create_harmonic_angle_dfdr ( double r ) {
	double r2 = r*r;
	if ( r2 == 1.0 )
		return -2.0 * potential_create_harmonic_angle_K;
	else
		return -2.0 * potential_create_harmonic_angle_K * ( acos(r) - potential_create_harmonic_angle_theta0 ) / sqrt( 1.0 - r2 );
}

double potential_create_harmonic_angle_d6fdr6 ( double r ) {
	return 0.0;
}

/**
 * @brief Creates a harmonic angle #potential
 *
 * @param a The smallest angle for which the potential will be constructed.
 * @param b The largest angle for which the potential will be constructed.
 * @param K The energy of the angle.
 * @param theta0 The minimum energy angle.
 * @param tol The tolerance to which the interpolation should match the exact
 *      potential.
 *
 * @return A newly-allocated #potential representing the potential
 *      @f$ K(\arccos(r)-r_0)^2 @f$ in @f$[a,b]@f$
 *      or @c NULL on error (see #potential_err).
 */

struct potential *potential_create_harmonic_angle ( double a , double b , double K , double theta0 , double tol ) {

	struct potential *p;
	double left, right;

	/* allocate the potential */
	if ( posix_memalign( (void **)&p , 16 , sizeof( struct potential ) ) != 0 ) {
		error(potential_err_malloc);
		return NULL;
	}

	/* Adjust a and b accordingly. */
	if ( a < 0.0 )
		a = 0.0;
	if ( b > M_PI )
		b = M_PI;
	left = cos(b);
	right = cos(a);
	if ( left - fabs(left)*sqrt(FPTYPE_EPSILON) < -1.0 )
		left = -1.0 / ( 1.0 + sqrt(FPTYPE_EPSILON) );
	if ( right + fabs(right)*sqrt(FPTYPE_EPSILON) > 1.0 )
		right = 1.0 / ( 1.0 + sqrt(FPTYPE_EPSILON) );

	/* fill this potential */
	potential_create_harmonic_angle_K = K;
	potential_create_harmonic_angle_theta0 = theta0;
	if ( potential_init( p , &potential_create_harmonic_angle_f , NULL , &potential_create_harmonic_angle_d6fdr6 , left , right , tol ) < 0 ) {
		free(p);
		return NULL;
	}

	/* return it */
			return p;

}


double potential_create_Ewald_q;
double potential_create_Ewald_kappa;

/* the potential functions */
double potential_create_Ewald_f ( double r ) {
	return potential_create_Ewald_q * potential_Ewald( r , potential_create_Ewald_kappa );
}

double potential_create_Ewald_dfdr ( double r ) {
	return potential_create_Ewald_q * potential_Ewald_p( r , potential_create_Ewald_kappa );
}

double potential_create_Ewald_d6fdr6 ( double r ) {
	return potential_create_Ewald_q * potential_Ewald_6p( r , potential_create_Ewald_kappa );
}

/**
 * @brief Creates a #potential representing the real-space part of an Ewald 
 *      potential.
 *
 * @param a The smallest radius for which the potential will be constructed.
 * @param b The largest radius for which the potential will be constructed.
 * @param q The charge scaling of the potential.
 * @param kappa The screening distance of the Ewald potential.
 * @param tol The tolerance to which the interpolation should match the exact
 *      potential.
 *
 * @return A newly-allocated #potential representing the potential
 *      @f$ q\frac{\mbox{erfc}(\kappa r}{r} @f$ in @f$[a,b]@f$
 *      or @c NULL on error (see #potential_err).
 */

struct potential *potential_create_Ewald ( double a , double b , double q , double kappa , double tol ) {

	struct potential *p;

	/* allocate the potential */
	if ( posix_memalign( (void **)&p , 16 , sizeof( struct potential ) ) != 0 ) {
		error(potential_err_malloc);
		return NULL;
	}

	/* fill this potential */
	potential_create_Ewald_q = q;
	potential_create_Ewald_kappa = kappa;
	if ( potential_init( p , &potential_create_Ewald_f , &potential_create_Ewald_dfdr , &potential_create_Ewald_d6fdr6 , a , b , tol ) < 0 ) {
		free(p);
		return NULL;
	}

	/* return it */
			return p;

}


double potential_create_LJ126_Ewald_A;
double potential_create_LJ126_Ewald_B;
double potential_create_LJ126_Ewald_kappa;
double potential_create_LJ126_Ewald_q;

/* the potential functions */
double potential_create_LJ126_Ewald_f ( double r ) {
	return potential_LJ126 ( r , potential_create_LJ126_Ewald_A , potential_create_LJ126_Ewald_B ) +
			potential_create_LJ126_Ewald_q * potential_Ewald( r , potential_create_LJ126_Ewald_kappa );
}

double potential_create_LJ126_Ewald_dfdr ( double r ) {
	return potential_LJ126_p ( r , potential_create_LJ126_Ewald_A , potential_create_LJ126_Ewald_B ) +
			potential_create_LJ126_Ewald_q * potential_Ewald_p( r , potential_create_LJ126_Ewald_kappa );
}

double potential_create_LJ126_Ewald_d6fdr6 ( double r ) {
	return potential_LJ126_6p ( r , potential_create_LJ126_Ewald_A , potential_create_LJ126_Ewald_B ) +
			potential_create_LJ126_Ewald_q * potential_Ewald_6p( r , potential_create_LJ126_Ewald_kappa );
}

/**
 * @brief Creates a #potential representing the sum of a
 *      12-6 Lennard-Jones potential and the real-space part of an Ewald 
 *      potential.
 *
 * @param a The smallest radius for which the potential will be constructed.
 * @param b The largest radius for which the potential will be constructed.
 * @param A The first parameter of the Lennard-Jones potential.
 * @param B The second parameter of the Lennard-Jones potential.
 * @param q The charge scaling of the potential.
 * @param kappa The screening distance of the Ewald potential.
 * @param tol The tolerance to which the interpolation should match the exact
 *      potential.
 *
 * @return A newly-allocated #potential representing the potential
 *      @f$ \left( \frac{A}{r^{12}} - \frac{B}{r^6} \right) @f$ in @f$[a,b]@f$
 *      or @c NULL on error (see #potential_err).
 */

struct potential *potential_create_LJ126_Ewald ( double a , double b , double A , double B , double q , double kappa , double tol ) {

	struct potential *p;

	/* allocate the potential */
	if ( posix_memalign( (void **)&p , 16 , sizeof( struct potential ) ) != 0 ) {
		error(potential_err_malloc);
		return NULL;
	}

	/* fill this potential */
	potential_create_LJ126_Ewald_A = A;
	potential_create_LJ126_Ewald_B = B;
	potential_create_LJ126_Ewald_kappa = kappa;
	potential_create_LJ126_Ewald_q = q;
	if ( potential_init( p , &potential_create_LJ126_Ewald_f , &potential_create_LJ126_Ewald_dfdr , &potential_create_LJ126_Ewald_d6fdr6 , a , b , tol ) < 0 ) {
		free(p);
		return NULL;
	}

	/* return it */
			return p;

}


double potential_create_LJ126_Ewald_switch_A;
double potential_create_LJ126_Ewald_switch_B;
double potential_create_LJ126_Ewald_switch_kappa;
double potential_create_LJ126_Ewald_switch_q;
double potential_create_LJ126_Ewald_switch_s;
double potential_create_LJ126_Ewald_switch_cutoff;

/* the potential functions */
double potential_create_LJ126_Ewald_switch_f ( double r ) {
	return potential_LJ126 ( r , potential_create_LJ126_Ewald_switch_A , potential_create_LJ126_Ewald_switch_B ) * potential_switch( r , potential_create_LJ126_Ewald_switch_s , potential_create_LJ126_Ewald_switch_cutoff ) +
			potential_create_LJ126_Ewald_switch_q * potential_Ewald( r , potential_create_LJ126_Ewald_switch_kappa );
}

double potential_create_LJ126_Ewald_switch_dfdr ( double r ) {
	return potential_LJ126_p ( r , potential_create_LJ126_Ewald_switch_A , potential_create_LJ126_Ewald_switch_B ) * potential_switch( r , potential_create_LJ126_Ewald_switch_s , potential_create_LJ126_Ewald_switch_cutoff ) +
			potential_LJ126 ( r , potential_create_LJ126_Ewald_switch_A , potential_create_LJ126_Ewald_switch_B ) * potential_switch_p( r , potential_create_LJ126_Ewald_switch_s , potential_create_LJ126_Ewald_switch_cutoff ) +
			potential_create_LJ126_Ewald_switch_q * potential_Ewald_p( r , potential_create_LJ126_Ewald_switch_kappa );
}

double potential_create_LJ126_Ewald_switch_d6fdr6 ( double r ) {
	return potential_LJ126_6p ( r , potential_create_LJ126_Ewald_switch_A , potential_create_LJ126_Ewald_switch_B ) +
			potential_create_LJ126_Ewald_switch_q * potential_Ewald_6p( r , potential_create_LJ126_Ewald_switch_kappa );
}

/**
 * @brief Creates a #potential representing the sum of a
 *      12-6 Lennard-Jones potential with a switching distance
 *      and the real-space part of an Ewald potential.
 *
 * @param a The smallest radius for which the potential will be constructed.
 * @param b The largest radius for which the potential will be constructed.
 * @param A The first parameter of the Lennard-Jones potential.
 * @param B The second parameter of the Lennard-Jones potential.
 * @param q The charge scaling of the potential.
 * @param s The switching distance.
 * @param kappa The screening distance of the Ewald potential.
 * @param tol The tolerance to which the interpolation should match the exact
 *      potential.
 *
 * @return A newly-allocated #potential representing the potential
 *      @f$ \left( \frac{A}{r^{12}} - \frac{B}{r^6} \right) @f$ in @f$[a,b]@f$
 *      or @c NULL on error (see #potential_err).
 */

struct potential *potential_create_LJ126_Ewald_switch ( double a , double b , double A , double B , double q , double kappa , double s , double tol ) {

	struct potential *p;

	/* allocate the potential */
	if ( posix_memalign( (void **)&p , 16 , sizeof( struct potential ) ) != 0 ) {
		error(potential_err_malloc);
		return NULL;
	}

	/* fill this potential */
	potential_create_LJ126_Ewald_switch_A = A;
	potential_create_LJ126_Ewald_switch_B = B;
	potential_create_LJ126_Ewald_switch_kappa = kappa;
	potential_create_LJ126_Ewald_switch_q = q;
	potential_create_LJ126_Ewald_switch_s = s;
	potential_create_LJ126_Ewald_switch_cutoff = b;
	if ( potential_init( p , &potential_create_LJ126_Ewald_switch_f , &potential_create_LJ126_Ewald_switch_dfdr , &potential_create_LJ126_Ewald_switch_d6fdr6 , a , b , tol ) < 0 ) {
		free(p);
		return NULL;
	}

	/* return it */
			return p;

}


double potential_create_Coulomb_q;
double potential_create_Coulomb_b;

/* the potential functions */
double potential_create_Coulomb_f ( double r ) {
	return potential_escale * potential_create_Coulomb_q * ( 1.0/r - 1.0/potential_create_Coulomb_b );
}

double potential_create_Coulomb_dfdr ( double r ) {
	return -potential_escale * potential_create_Coulomb_q / ( r * r );
}

double potential_create_Coulomb_d6fdr6 ( double r ) {
	double r2 = r*r, r4 = r2*r2, r7 = r*r2*r4;
	return 720.0 * potential_escale * potential_create_Coulomb_q / r7;
}

/**
 * @brief Creates a #potential representing a shifted Coulomb potential.
 *
 * @param a The smallest radius for which the potential will be constructed.
 * @param b The largest radius for which the potential will be constructed.
 * @param q The charge scaling of the potential.
 * @param tol The tolerance to which the interpolation should match the exact
 *      potential.
 *
 * @return A newly-allocated #potential representing the potential
 *      @f$ \frac{1}{4\pi r} @f$ in @f$[a,b]@f$
 *      or @c NULL on error (see #potential_err).
 */

struct potential *potential_create_Coulomb ( double a , double b , double q , double tol ) {

	struct potential *p;

	/* allocate the potential */
	if ( posix_memalign( (void **)&p , 16 , sizeof( struct potential ) ) != 0 ) {
		error(potential_err_malloc);
		return NULL;
	}

	/* fill this potential */
	potential_create_Coulomb_q = q;
	potential_create_Coulomb_b = b;
	if ( potential_init( p , &potential_create_Coulomb_f , &potential_create_Coulomb_dfdr , &potential_create_Coulomb_d6fdr6 , a , b , tol ) < 0 ) {
		free(p);
		return NULL;
	}

	/* return it */
			return p;

}


double potential_create_LJ126_Coulomb_q;
double potential_create_LJ126_Coulomb_b;
double potential_create_LJ126_Coulomb_A;
double potential_create_LJ126_Coulomb_B;

/* the potential functions */
double potential_create_LJ126_Coulomb_f ( double r ) {
	return potential_LJ126 ( r , potential_create_LJ126_Coulomb_A , potential_create_LJ126_Coulomb_B ) +
			potential_escale * potential_create_LJ126_Coulomb_q * ( 1.0/r - 1.0/potential_create_LJ126_Coulomb_b );
}

double potential_create_LJ126_Coulomb_dfdr ( double r ) {
	return potential_LJ126_p ( r , potential_create_LJ126_Coulomb_A , potential_create_LJ126_Coulomb_B ) -
			potential_escale * potential_create_LJ126_Coulomb_q / ( r * r );
}

double potential_create_LJ126_Coulomb_d6fdr6 ( double r ) {
	double r2 = r*r, r4 = r2*r2, r7 = r*r2*r4;
	return potential_LJ126_6p ( r , potential_create_LJ126_Coulomb_A , potential_create_LJ126_Coulomb_B ) +
			720.0 * potential_escale * potential_create_LJ126_Coulomb_q / r7;
}

/**
 * @brief Creates a #potential representing the sum of a
 *      12-6 Lennard-Jones potential and a shifted Coulomb potential.
 *
 * @param a The smallest radius for which the potential will be constructed.
 * @param b The largest radius for which the potential will be constructed.
 * @param A The first parameter of the Lennard-Jones potential.
 * @param B The second parameter of the Lennard-Jones potential.
 * @param q The charge scaling of the potential.
 * @param tol The tolerance to which the interpolation should match the exact
 *      potential.
 *
 * @return A newly-allocated #potential representing the potential
 *      @f$ \left( \frac{A}{r^{12}} - \frac{B}{r^6} \right) @f$ in @f$[a,b]@f$
 *      or @c NULL on error (see #potential_err).
 */

struct potential *potential_create_LJ126_Coulomb ( double a , double b , double A , double B , double q , double tol ) {

	struct potential *p;

	/* allocate the potential */
	if ( posix_memalign( (void **)&p , 16 , sizeof( struct potential ) ) != 0 ) {
		error(potential_err_malloc);
		return NULL;
	}

	/* fill this potential */
	potential_create_LJ126_Coulomb_q = q;
	potential_create_LJ126_Coulomb_b = b;
	potential_create_LJ126_Coulomb_A = A;
	potential_create_LJ126_Coulomb_B = B;
	if ( potential_init( p , &potential_create_LJ126_Coulomb_f , &potential_create_LJ126_Coulomb_dfdr , &potential_create_LJ126_Coulomb_d6fdr6 , a , b , tol ) < 0 ) {
		free(p);
		return NULL;
	}

	/* return it */
			return p;

}


double potential_create_LJ126_A;
double potential_create_LJ126_B;

/* the potential functions */
double potential_create_LJ126_f ( double r ) {
	return potential_LJ126 ( r , potential_create_LJ126_A , potential_create_LJ126_B );
}

double potential_create_LJ126_dfdr ( double r ) {
	return potential_LJ126_p ( r , potential_create_LJ126_A , potential_create_LJ126_B );
}

double potential_create_LJ126_d6fdr6 ( double r ) {
	return potential_LJ126_6p ( r , potential_create_LJ126_A , potential_create_LJ126_B );
}

/**
 * @brief Creates a #potential representing a 12-6 Lennard-Jones potential
 *
 * @param a The smallest radius for which the potential will be constructed.
 * @param b The largest radius for which the potential will be constructed.
 * @param A The first parameter of the Lennard-Jones potential.
 * @param B The second parameter of the Lennard-Jones potential.
 * @param tol The tolerance to which the interpolation should match the exact
 *      potential.
 *
 * @return A newly-allocated #potential representing the potential
 *      @f$ \left( \frac{A}{r^{12}} - \frac{B}{r^6} \right) @f$ in @f$[a,b]@f$
 *      or @c NULL on error (see #potential_err).
 *
 */

struct potential *potential_create_LJ126 ( double a , double b , double A , double B , double tol ) {

	struct potential *p;

	/* allocate the potential */
	if ( posix_memalign( (void **)&p , 16 , sizeof( struct potential ) ) != 0 ) {
		error(potential_err_malloc);
		return NULL;
	}

	/* fill this potential */
	potential_create_LJ126_A = A;
	potential_create_LJ126_B = B;
	if ( potential_init( p , &potential_create_LJ126_f , &potential_create_LJ126_dfdr , &potential_create_LJ126_d6fdr6 , a , b , tol ) < 0 ) {
		free(p);
		return NULL;
	}

	/* return it */
			return p;

}


double potential_create_LJ126_switch_A;
double potential_create_LJ126_switch_B;
double potential_create_LJ126_switch_s;
double potential_create_LJ126_switch_cutoff;

/* the potential functions */
double potential_create_LJ126_switch_f ( double r ) {
	return potential_LJ126( r , potential_create_LJ126_switch_A , potential_create_LJ126_switch_B ) * potential_switch( r , potential_create_LJ126_switch_s , potential_create_LJ126_switch_cutoff );
}

double potential_create_LJ126_switch_dfdr ( double r ) {
	return potential_LJ126_p ( r , potential_create_LJ126_switch_A , potential_create_LJ126_switch_B ) * potential_switch( r , potential_create_LJ126_switch_s , potential_create_LJ126_switch_cutoff ) +
			potential_LJ126( r , potential_create_LJ126_switch_A , potential_create_LJ126_switch_B ) * potential_switch_p( r , potential_create_LJ126_switch_s , potential_create_LJ126_switch_cutoff );
}

double potential_create_LJ126_switch_d6fdr6 ( double r ) {
	return potential_LJ126_6p( r , potential_create_LJ126_switch_A , potential_create_LJ126_switch_B );
}

/**
 * @brief Creates a #potential representing a switched 12-6 Lennard-Jones potential
 *
 * @param a The smallest radius for which the potential will be constructed.
 * @param b The largest radius for which the potential will be constructed.
 * @param A The first parameter of the Lennard-Jones potential.
 * @param B The second parameter of the Lennard-Jones potential.
 * @param s The switchting length
 * @param tol The tolerance to which the interpolation should match the exact
 *      potential.
 *
 * @return A newly-allocated #potential representing the potential
 *      @f$ \left( \frac{A}{r^{12}} - \frac{B}{r^6} \right) @f$ in @f$[a,b]@f$
 *      or @c NULL on error (see #potential_err).
 *
 */

struct potential *potential_create_LJ126_switch ( double a , double b , double A , double B , double s , double tol ) {

	struct potential *p;

	/* allocate the potential */
	if ( posix_memalign( (void **)&p , 16 , sizeof( struct potential ) ) != 0 ) {
		error(potential_err_malloc);
		return NULL;
	}

	/* fill this potential */
	potential_create_LJ126_switch_A = A;
	potential_create_LJ126_switch_B = B;
	potential_create_LJ126_switch_s = s;
	potential_create_LJ126_switch_cutoff = b;
	if ( potential_init( p , &potential_create_LJ126_switch_f , &potential_create_LJ126_switch_dfdr , &potential_create_LJ126_switch_d6fdr6 , a , b , tol ) < 0 ) {
		free(p);
		return NULL;
	}

	/* return it */
			return p;

}


/**
 * @brief Free the memory associated with the given potential.
 * 
 * @param p Pointer to the #potential to clear.
 */

void potential_clear ( struct potential *p ) {

	/* Do nothing? */
	if ( p == NULL )
		return;

	/* Clear the flags. */
	p->flags = potential_flag_none;

	/* Clear the coefficients. */
	free( p->c );
	p->c = NULL;

}


/**
 * @brief Construct a #potential from the given function.
 *
 * @param p A pointer to an empty #potential.
 * @param f A pointer to the potential function to be interpolated.
 * @param fp A pointer to the first derivative of @c f.
 * @param f6p A pointer to the sixth derivative of @c f.
 * @param a The smallest radius for which the potential will be constructed.
 * @param b The largest radius for which the potential will be constructed.
 * @param tol The absolute tolerance to which the interpolation should match
 *      the exact potential.
 *
 * @return #potential_err_ok or <0 on error (see #potential_err).
 *
 * Computes an interpolated potential function from @c f in @c [a,b] to the
 * locally relative tolerance @c tol.
 *
 * The sixth derivative @c f6p is used to compute the optimal node
 * distribution. If @c f6p is @c NULL, the derivative is approximated
 * numerically.
 *
 * The zeroth interval contains a linear extension of @c f for values < a.
 */

int potential_init ( struct potential *p , double (*f)( double ) , double (*fp)( double ) , double (*f6p)( double ) , FPTYPE a , FPTYPE b , FPTYPE tol ) {

	double alpha, w;
	int l = potential_ivalsa, r = potential_ivalsb, m;
	FPTYPE err_l, err_r, err_m;
	FPTYPE *xi_l, *xi_r, *xi_m;
	FPTYPE *c_l, *c_r, *c_m;
	int i, k;
	double e;
	FPTYPE mtol = 10 * FPTYPE_EPSILON;

	/* check inputs */
	if ( p == NULL || f == NULL )
		return error(potential_err_null);

	/* check if we have a user-specified 6th derivative or not. */
	if ( f6p == NULL )
		return error(potential_err_nyi);

	/* Stretch the domain ever so slightly to accommodate for rounding
       error when computing the index. */
	b += fabs(b) * sqrt(FPTYPE_EPSILON);
	a -= fabs(a) * sqrt(FPTYPE_EPSILON);
	// printf( "potential_init: setting a=%.16e, b=%.16e.\n" , a , b );

	/* set the boundaries */
	p->a = a; p->b = b;

	/* compute the optimal alpha for this potential */
	alpha = potential_getalpha(f6p,a,b);
	/* printf("potential_init: alpha is %22.16e\n",(double)alpha); fflush(stdout); */

	/* compute the interval transform */
	w = 1.0 / (a - b); w *= w;
	p->alpha[0] = a*a*w - alpha*b*a*w;
	p->alpha[1] = -2*a*w + alpha*(a+b)*w;
	p->alpha[2] = w - alpha*w;
	p->alpha[3] = 0.0;

	/* Correct the transform to the right. */
	w = 2*FPTYPE_EPSILON*(fabs(p->alpha[0])+fabs(p->alpha[1])+fabs(p->alpha[2]));
	p->alpha[0] -= w*a/(a-b);
	p->alpha[1] += w/(a-b);

	/* compute the smallest interpolation... */
	/* printf("potential_init: trying l=%i...\n",l); fflush(stdout); */
	xi_l = (FPTYPE *)malloc( sizeof(FPTYPE) * (l + 1) );
	c_l = (FPTYPE *)malloc( sizeof(FPTYPE) * (l+1) * potential_chunk );
	if ( posix_memalign( (void **)&c_l , potential_align , sizeof(FPTYPE) * (l+1) * potential_chunk ) < 0 )
		return error(potential_err_malloc);
	xi_l[0] = a; xi_l[l] = b;
	for ( i = 1 ; i < l ; i++ ) {
		xi_l[i] = a + (b - a) * i / l;
		while ( 1 ) {
			e = i - l * (p->alpha[0] + xi_l[i]*(p->alpha[1] + xi_l[i]*p->alpha[2]));
			xi_l[i] += e / (l * (p->alpha[1] + 2*xi_l[i]*p->alpha[2]));
			if ( fabs(e) < l*mtol )
				break;
		}
	}
	if ( potential_getcoeffs(f,fp,xi_l,l,&c_l[potential_chunk],&err_l) < 0 )
		return error(potential_err);
	/* fflush(stderr); printf("potential_init: err_l=%22.16e.\n",err_l); */

	/* if this interpolation is good enough, stop here! */
	if ( err_l < tol ) {

		/* Set the domain variables. */
		p->n = l;
		p->c = c_l;
		p->alpha[0] *= p->n; p->alpha[1] *= p->n; p->alpha[2] *= p->n;
		p->alpha[0] += 1;

		/* Fix the first interval. */
		p->c[0] = a; p->c[1] = 1.0 / a;
		double coeffs[potential_degree], eff[potential_degree];
		for ( k = 0 ; k < potential_degree ; k++ ) {
			coeffs[k] = p->c[2*potential_chunk-1-k];
			eff[k] = 0.0;
		}
		for ( i = 0 ; i < potential_degree ; i++ )
			for ( k = potential_degree-1 ; k >= i ; k-- ) {
				eff[i] = coeffs[k] + (-1.0)*eff[i];
				coeffs[k] *= (k - i) * p->c[potential_chunk+1] * a;
			}
		p->c[potential_chunk-1] = eff[0];
		p->c[potential_chunk-2] = eff[1];
		p->c[potential_chunk-3] = 0.5 * eff[2];
		// p->c[potential_chunk-4] = (eff[2] - eff[1] ) / 3;
		for ( k = 3 ; k <= potential_degree ; k++ )
			p->c[potential_chunk-1-k] = 0.0;

		/* Clean up. */
		free(xi_l);

		return potential_err_ok;
	}

	/* loop until we have an upper bound on the right... */
	while ( 1 ) {

		/* compute the larger interpolation... */
		/* printf("potential_init: trying r=%i...\n",r); fflush(stdout); */
		xi_r = (FPTYPE *)malloc( sizeof(FPTYPE) * (r + 1) );
		if ( posix_memalign( (void **)&c_r , potential_align , sizeof(FPTYPE) * (r+1) * potential_chunk ) != 0 )
			return error(potential_err_malloc);
		xi_r[0] = a; xi_r[r] = b;
		for ( i = 1 ; i < r ; i++ ) {
			xi_r[i] = a + (b - a) * i / r;
			while ( 1 ) {
				e = i - r * (p->alpha[0] + xi_r[i]*(p->alpha[1] + xi_r[i]*p->alpha[2]));
				xi_r[i] += e / (r * (p->alpha[1] + 2*xi_r[i]*p->alpha[2]));
				if ( fabs(e) < r*mtol )
					break;
			}
		}
		if ( potential_getcoeffs(f,fp,xi_r,r,&c_r[potential_chunk],&err_r) < 0 )
			return error(potential_err);
		/* printf("potential_init: err_r=%22.16e.\n",err_r); fflush(stdout); */

		/* if this is better than tolerance, break... */
		if ( err_r < tol )
			break;

		/* Have we too many intervals? */
		else if ( 2*r > potential_ivalsmax ) {
			/* printf( "potential_init: warning: maximum nr of intervals (%i) reached, err=%e.\n" , r , err_r );
            break; */
			return error(potential_err_ivalsmax);
		}

		/* otherwise, l=r and r = 2*r */
		else {
			l = r; err_l = err_r;
			free(xi_l); xi_l = xi_r;
			free(c_l); c_l = c_r;
			r *= 2;
		}

	} /* loop until we have a good right estimate */

	/* we now have a left and right estimate -- binary search! */
	while ( r - l > 1 ) {

		/* find the middle */
		m = 0.5 * ( r + l );

		/* construct that interpolation */
		/* printf("potential_init: trying m=%i...\n",m); fflush(stdout); */
		xi_m = (FPTYPE *)malloc( sizeof(FPTYPE) * (m + 1) );
		if ( posix_memalign( (void **)&c_m , potential_align , sizeof(FPTYPE) * (m+1) * potential_chunk ) != 0 )
			return error(potential_err_malloc);
		xi_m[0] = a; xi_m[m] = b;
		for ( i = 1 ; i < m ; i++ ) {
			xi_m[i] = a + (b - a) * i / m;
			while ( 1 ) {
				e = i - m * (p->alpha[0] + xi_m[i]*(p->alpha[1] + xi_m[i]*p->alpha[2]));
				xi_m[i] += e / (m * (p->alpha[1] + 2*xi_m[i]*p->alpha[2]));
				if ( fabs(e) < m*mtol )
					break;
			}
		}
		if ( potential_getcoeffs(f,fp,xi_m,m,&c_m[potential_chunk],&err_m) != 0 )
			return error(potential_err);
		/* printf("potential_init: err_m=%22.16e.\n",err_m); fflush(stdout); */

		/* go left? */
				if ( err_m > tol ) {
					l = m; err_l = err_m;
					free(xi_l); xi_l = xi_m;
					free(c_l); c_l = c_m;
				}

				/* otherwise, go right... */
				else {
					r = m; err_r = err_m;
					free(xi_r); xi_r = xi_m;
					free(c_r); c_r = c_m;
				}

	} /* binary search */

	/* as of here, the right estimate is the smallest interpolation below */
	/* the requested tolerance */
	p->n = r;
	p->c = c_r;
	p->alpha[0] *= p->n; p->alpha[1] *= p->n; p->alpha[2] *= p->n;
	p->alpha[0] += 1.0;

	/* Make the first interval a linear continuation. */
	p->c[0] = a; p->c[1] = 1.0 / a;
	double coeffs[potential_degree], eff[potential_degree];
	for ( k = 0 ; k < potential_degree ; k++ ) {
		coeffs[k] = p->c[2*potential_chunk-1-k];
		eff[k] = 0.0;
	}
	for ( i = 0 ; i < potential_degree ; i++ )
		for ( k = potential_degree-1 ; k >= i ; k-- ) {
			eff[i] = coeffs[k] + (-1.0)*eff[i];
			coeffs[k] *= (k - i) * p->c[potential_chunk+1] * a;
		}
	p->c[potential_chunk-1] = eff[0];
	p->c[potential_chunk-2] = eff[1];
	p->c[potential_chunk-3] = 0.5 * eff[2];
	// p->c[potential_chunk-4] = (eff[2] - eff[1] ) / 3;
	for ( k = 3 ; k <= potential_degree ; k++ )
		p->c[potential_chunk-1-k] = 0.0;

	/* Clean up. */
	free(xi_r);
	free(xi_l); free(c_l);

	/* all is well that ends well... */
	return potential_err_ok;

}


/**
 * @brief Compute the optimal first derivatives for the given set of
 *      nodes.
 *
 * @param f Pointer to the function to be interpolated.
 * @param n Number of intervals.
 * @param xi Pointer to an array of nodes between whicht the function @c f
 *      will be interpolated.
 * @param fp Pointer to an array in which to store the first derivatives
 *      of @c f.
 *
 * @return #potential_err_ok or < 0 on error (see #potential_err).
 */

int potential_getfp ( double (*f)( double ) , int n , FPTYPE *x , double *fp ) {

	int i, k;
	double m, h, eff, fx[n+1], hx[n];
	double d0[n+1], d1[n+1], d2[n+1], b[n+1];
	double viwl1[n], viwr1[n];
	static double *w = NULL, *xi = NULL;

	/* Cardinal functions. */
	const double cwl1[4] = { 0.25, -0.25, -0.25, 0.25 };
	const double cwr1[4] = { -0.25, -0.25, 0.25, 0.25 };
	const double wl0wl1 = 0.1125317885884428;
	const double wl1wl1 = 0.03215579530433858;
	const double wl0wr1 = -0.04823369227661384;
	const double wl1wr1 = -0.02143719641629633;
	const double wr0wr1 = -0.1125317885884429;
	const double wr1wr1 = 0.03215579530433859;
	const double wl1wr0 = 0.04823369227661384;

	/* Pre-compute the weights? */
	if ( w == NULL ) {
		if ( ( w = (double *)malloc( sizeof(double) * potential_N ) ) == NULL ||
				( xi = (double *)malloc( sizeof(double) * potential_N ) ) == NULL )
			return error(potential_err_malloc);
		for ( k = 1 ; k < potential_N-1 ; k++ ) {
			xi[k] = cos( k * M_PI / (potential_N - 1) );
			w[k] = 1.0 / sqrt( 1.0 - xi[k]*xi[k] );
		}
		xi[0] = 1.0; xi[potential_N-1] = -1.0;
		w[0] = 0.0; w[potential_N-1] = 0.0;
	}

	/* Get the values of fx and ih. */
	for ( i = 0 ; i <= n ; i++ )
		fx[i] = f( x[i] );
	for ( i = 0 ; i < n ; i++ )
		hx[i] = x[i+1] - x[i];

	/* Compute the products of f with respect to wl1 and wr1. */
	for ( i = 0 ; i < n ; i++ ) {
		viwl1[i] = 0.0; viwr1[i] = 0.0;
		m = 0.5*(x[i] + x[i+1]);
		h = 0.5*(x[i+1] - x[i]);
		for ( k = 1 ; k < potential_N-1 ; k++ ) {
			eff = f( m + h*xi[k] );
			viwl1[i] += w[k] * ( eff * ( cwl1[0] + xi[k]*(cwl1[1] + xi[k]*(cwl1[2] + xi[k]*cwl1[3])) ) );
			viwr1[i] += w[k] * ( eff * ( cwr1[0] + xi[k]*(cwr1[1] + xi[k]*(cwr1[2] + xi[k]*cwr1[3])) ) );
		}
		viwl1[i] /= potential_N-2;
		viwr1[i] /= potential_N-2;
	}

	/* Fill the diagonals and the right-hand side. */
	d1[0] = wl1wl1 * hx[0];
	d2[0] = wl1wr1 * hx[0];
	b[0] = 2 * ( viwl1[0] - fx[0]*wl0wl1 - fx[1]*wl1wr0 );
	for ( i = 1 ; i < n ; i++ ) {
		d0[i] = wl1wr1 * hx[i-1];
		d1[i] = wr1wr1 * hx[i-1] + wl1wl1 * hx[i];
		d2[i] = wl1wr1 * hx[i];
		b[i] = 2 * ( viwr1[i-1] - fx[i-1]*wl0wr1 - fx[i]*wr0wr1 ) +
				2 * ( viwl1[i] - fx[i]*wl0wl1 - fx[i+1]*wl1wr0 );
	}
	d0[n] = wl1wr1 * hx[n-1];
	d1[n] = wr1wr1 * hx[n-1];
	b[n] = 2 * ( viwr1[n-1] - fx[n-1]*wl0wr1 - fx[n]*wr0wr1 );

	/* Solve the trilinear system. */
	for ( i = 1 ; i <= n ; i++ )  {
		m = d0[i]/d1[i-1];
		d1[i] = d1[i] - m*d2[i-1];
		b[i] = b[i] - m*b[i-1];
	}
	fp[n] = b[n]/d1[n];
	for ( i = n - 1 ; i >= 0 ; i-- )
		fp[i] = ( b[i] - d2[i]*fp[i+1] ) / d1[i];

	/* Fingers crossed... */
	return potential_err_ok;

}


int potential_getfp_fixend ( double (*f)( double ) , double fpa , double fpb , int n , FPTYPE *x , double *fp ) {

	int i, k;
	double m, h, eff, fx[n+1], hx[n];
	double d0[n+1], d1[n+1], d2[n+1], b[n+1];
	double viwl1[n], viwr1[n];
	static double *w = NULL, *xi = NULL;

	/* Cardinal functions. */
	const double cwl1[4] = { 0.25, -0.25, -0.25, 0.25 };
	const double cwr1[4] = { -0.25, -0.25, 0.25, 0.25 };
	const double wl0wl1 = 0.1125317885884428;
	const double wl1wl1 = 0.03215579530433858;
	const double wl0wr1 = -0.04823369227661384;
	const double wl1wr1 = -0.02143719641629633;
	const double wr0wr1 = -0.1125317885884429;
	const double wr1wr1 = 0.03215579530433859;
	const double wl1wr0 = 0.04823369227661384;

	/* Pre-compute the weights? */
	if ( w == NULL ) {
		if ( ( w = (double *)malloc( sizeof(double) * potential_N ) ) == NULL ||
				( xi = (double *)malloc( sizeof(double) * potential_N ) ) == NULL )
			return error(potential_err_malloc);
		for ( k = 1 ; k < potential_N-1 ; k++ ) {
			xi[k] = cos( k * M_PI / (potential_N - 1) );
			w[k] = 1.0 / sqrt( 1.0 - xi[k]*xi[k] );
		}
		xi[0] = 1.0; xi[potential_N-1] = -1.0;
		w[0] = 0.0; w[potential_N-1] = 0.0;
	}

	/* Get the values of fx and ih. */
	for ( i = 0 ; i <= n ; i++ )
		fx[i] = f( x[i] );
	for ( i = 0 ; i < n ; i++ )
		hx[i] = x[i+1] - x[i];

	/* Compute the products of f with respect to wl1 and wr1. */
	for ( i = 0 ; i < n ; i++ ) {
		viwl1[i] = 0.0; viwr1[i] = 0.0;
		m = 0.5*(x[i] + x[i+1]);
		h = 0.5*(x[i+1] - x[i]);
		for ( k = 1 ; k < potential_N-1 ; k++ ) {
			eff = f( m + h*xi[k] );
			viwl1[i] += w[k] * ( eff * ( cwl1[0] + xi[k]*(cwl1[1] + xi[k]*(cwl1[2] + xi[k]*cwl1[3])) ) );
			viwr1[i] += w[k] * ( eff * ( cwr1[0] + xi[k]*(cwr1[1] + xi[k]*(cwr1[2] + xi[k]*cwr1[3])) ) );
		}
		viwl1[i] /= potential_N-2;
		viwr1[i] /= potential_N-2;
	}

	/* Fill the diagonals and the right-hand side. */
	d1[0] = 1.0;
	d2[0] = 0.0;
	b[0] = fpa;
	for ( i = 1 ; i < n ; i++ ) {
		d0[i] = wl1wr1 * hx[i-1];
		d1[i] = wr1wr1 * hx[i-1] + wl1wl1 * hx[i];
		d2[i] = wl1wr1 * hx[i];
		b[i] = 2 * ( viwr1[i-1] - fx[i-1]*wl0wr1 - fx[i]*wr0wr1 ) +
				2 * ( viwl1[i] - fx[i]*wl0wl1 - fx[i+1]*wl1wr0 );
	}
	d0[n] = 0.0;
	d1[n] = 1.0;
	b[n] = fpb;

	/* Solve the trilinear system. */
	for ( i = 1 ; i <= n ; i++ )  {
		m = d0[i]/d1[i-1];
		d1[i] = d1[i] - m*d2[i-1];
		b[i] = b[i] - m*b[i-1];
	}
	fp[n] = b[n]/d1[n];
	for ( i = n - 1 ; i >= 0 ; i-- )
		fp[i] = ( b[i] - d2[i]*fp[i+1] ) / d1[i];

	/* Fingers crossed... */
	return potential_err_ok;

}


/**
 * @brief Compute the interpolation coefficients over a given set of nodes.
 * 
 * @param f Pointer to the function to be interpolated.
 * @param fp Pointer to the first derivative of @c f.
 * @param xi Pointer to an array of nodes between whicht the function @c f
 *      will be interpolated.
 * @param n Number of nodes in @c xi.
 * @param c Pointer to an array in which to store the interpolation
 *      coefficients.
 * @param err Pointer to a floating-point value in which an approximation of
 *      the interpolation error, relative to the maximum of f in each interval,
 *      is stored.
 *
 * @return #potential_err_ok or < 0 on error (see #potential_err).
 *
 * Compute the coefficients of the function @c f with derivative @c fp
 * over the @c n intervals between the @c xi and store an estimate of the
 * maximum locally relative interpolation error in @c err.
 *
 * The array to which @c c points must be large enough to hold at least
 * #potential_degree x @c n values of type #FPTYPE.
 */

int potential_getcoeffs ( double (*f)( double ) , double (*fp)( double ) , FPTYPE *xi , int n , FPTYPE *c , FPTYPE *err ) {

	// TODO, seriously buggy shit here!
	// make sure all arrays are of length n+1
	int i, j, k, ind;
	double phi[7], cee[6], fa, fb, dfa, dfb, fix[n+1], fpx[n+1];
	double h, m, w, e, err_loc, maxf, x;
	double fx[potential_N];
	static FPTYPE *coskx = NULL;

	/* check input sanity */
	if ( f == NULL || xi == NULL || err == NULL )
		return error(potential_err_null);

	/* Do we need to init the pre-computed cosines? */
	if ( coskx == NULL ) {
		if ( ( coskx = (FPTYPE *)malloc( sizeof(FPTYPE) * 7 * potential_N ) ) == NULL )
			return error(potential_err_malloc);
		for ( k = 0 ; k < 7 ; k++ )
			for ( j = 0 ; j < potential_N ; j++ )
				coskx[ k*potential_N + j ] = cos( j * k * M_PI / potential_N );
	}

	/* Get fx and fpx. */
	for ( k = 0 ; k <= n ; k++ ) {
		fix[k] = f( xi[k] );
		// fpx[k] = fp( xi[k] );
	}

	/* Compute the optimal fpx. */
	if ( fp == NULL ) {
		if ( potential_getfp( f , n , xi , fpx ) < 0 )
			return error(potential_err);
	}
	else {
		if ( potential_getfp_fixend( f , fp(xi[0]) , fp(xi[n]) , n , xi , fpx ) < 0 )
			return error(potential_err);
	}
	/* for ( k = 0 ; k <= n ; k++ )
        printf( "potential_getcoeffs: fp[%i]=%e , fpx[%i]=%e.\n" , k , fp(xi[k]) , k , fpx[k] );
    fflush(stdout); getchar(); */

	/* init the maximum interpolation error */
	*err = 0.0;

	/* loop over all intervals... */
	for ( i = 0 ; i < n ; i++ ) {

		/* set the initial index */
		ind = i * (potential_degree + 3);

		/* get the interval centre and width */
		m = (xi[i] + xi[i+1]) / 2;
		h = (xi[i+1] - xi[i]) / 2;

		/* evaluate f and fp at the edges */
		fa = fix[i]; fb = fix[i+1];
		dfa = fpx[i] * h; dfb = fpx[i+1] * h;
		// printf("potential_getcoeffs: xi[i]=%22.16e\n",xi[i]);

		/* compute the coefficients phi of f */
		for ( k = 0 ; k < potential_N ; k++ )
			fx[k] = f( m + h * cos( k * M_PI / potential_N ) );
		for ( j = 0 ; j < 7 ; j++ ) {
			phi[j] = (fa + (1-2*(j%2))*fb) / 2;
			for ( k = 1 ; k < potential_N ; k++ )
				phi[j] += fx[k] * coskx[ j*potential_N + k ];
			phi[j] *= 2.0 / potential_N;
		}

		/* compute the first four coefficients */
		cee[0] = (4*(fa + fb) + dfa - dfb) / 4;
		cee[1] = -(9*(fa - fb) + dfa + dfb) / 16;
		cee[2] = (dfb - dfa) / 8;
		cee[3] = (fa - fb + dfa + dfb) / 16;
		cee[4] = 0.0;
		cee[5] = 0.0;

		/* add the 4th correction... */
		w = ( 6 * ( cee[0] - phi[0]) - 4 * ( cee[2] - phi[2] ) - phi[4] ) / ( 36 + 16 + 1 );
		cee[0] += -6 * w;
		cee[2] += 4 * w;
		cee[4] = -w;

		/* add the 5th correction... */
		w = ( 2 * ( cee[1] - phi[1]) - 3 * ( cee[3] - phi[3] ) - phi[5] ) / ( 4 + 9 + 1 );
		cee[1] += -2 * w;
		cee[3] += 3 * w;
		cee[5] = -w;

		/* convert to monomials on the interval [-1,1] */
		c[ind+7] = cee[0]/2 - cee[2] + cee[4];
		c[ind+6] = cee[1] - 3*cee[3] + 5*cee[5];
		c[ind+5] = 2*cee[2] - 8*cee[4];
		c[ind+4] = 4*cee[3] - 20*cee[5];
		c[ind+3] = 8*cee[4];
		c[ind+2] = 16*cee[5];
		c[ind+1] = 1.0 / h;
		c[ind] = m;

		/* compute a local error estimate (klutzy) */
		maxf = 0.0; err_loc = 0.0;
		for ( k = 1 ; k < potential_N ; k++ ) {
			maxf = fmax( fabs( fx[k] ) , maxf );
			x = coskx[ potential_N + k ];
			e = fabs( fx[k] - c[ind+7]
								-x * ( c[ind+6] +
										x * ( c[ind+5] +
												x * ( c[ind+4] +
														x * ( c[ind+3] +
																x * c[ind+2] )))) );
			err_loc = fmax( e , err_loc );
		}
		err_loc /= fmax( maxf , 1.0 );
		*err = fmax( err_loc , *err );

	}

	/* all is well that ends well... */
	return potential_err_ok;

}


/**
 * @brief Compute the parameter @f$\alpha@f$ for the optimal node distribution.
 *
 * @param f6p Pointer to a function representing the 6th derivative of the
 *      interpoland.
 * @param a Left limit of the interpolation.
 * @param b Right limit of the interpolation.
 *
 * @return The computed value for @f$\alpha@f$.
 *
 * The value @f$\alpha@f$ is computed using Brent's algortihm to 4 decimal
 * digits.
 */

double potential_getalpha ( double (*f6p)( double ) , double a , double b ) {

	double xi[potential_N], fx[potential_N];
	int i, j;
	double temp;
	double alpha[4], fa[4], maxf = 0.0;
	const double golden = 2.0 / (1 + sqrt(5));

	/* start by evaluating f6p at the N nodes between 'a' and 'b' */
	for ( i = 0 ; i < potential_N ; i++ ) {
		xi[i] = ((double)i + 1) / (potential_N + 1);
		fx[i] = f6p( a + (b-a) * xi[i] );
		maxf = fmax( maxf , fabs(fx[i]) );
	}

	/* Trivial? */
	if ( maxf == 0.0 )
		return 1.0;

	/* set the initial values for alpha */
	alpha[0] = 0; alpha[3] = 2;
	alpha[1] = alpha[3] - 2 * golden; alpha[2] = alpha[0] + 2 * golden;
	for ( i = 0 ; i < 4 ; i++ ) {
		fa[i] = 0.0;
		for ( j = 0 ; j < potential_N ; j++ ) {
			temp = fabs( pow( alpha[i] + 2 * (1 - alpha[i]) * xi[j] , -6 ) * fx[j] );
			if ( temp > fa[i] )
				fa[i] = temp;
		}
	}

	/* main loop (brent's algorithm) */
			while ( alpha[3] - alpha[0] > 1.0e-4 ) {

				/* go west? */
				if ( fa[1] < fa[2] ) {
					alpha[3] = alpha[2]; fa[3] = fa[2];
					alpha[2] = alpha[1]; fa[2] = fa[1];
					alpha[1] = alpha[3] - (alpha[3] - alpha[0]) * golden;
					i = 1;
				}

				/* nope, go east... */
				else {
					alpha[0] = alpha[1]; fa[0] = fa[1];
					alpha[1] = alpha[2]; fa[1] = fa[2];
					alpha[2] = alpha[0] + (alpha[3] - alpha[0]) * golden;
					i = 2;
				}

				/* compute the new value */
				fa[i] = 0.0;
				for ( j = 0 ; j < potential_N ; j++ ) {
					temp = fabs( pow( alpha[i] + 2 * (1 - alpha[i]) * xi[j] , -6 ) * fx[j] );
					if ( temp > fa[i] )
						fa[i] = temp;
				}

			} /* main loop */

	/* return the average */
	return (alpha[0] + alpha[3]) / 2;

}

