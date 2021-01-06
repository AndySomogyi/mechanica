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

#ifdef _WIN32
#define _USE_MATH_DEFINES
#endif

/* include some standard header files */
#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <MxPotential.h>
#include <MxParticle.h>
#include <MxPy.h>
#include <string.h>
#include <carbon.h>
#include <CConvert.hpp>


/* include local headers */
#include "errs.h"
#include "fptype.h"
#include "potential_eval.h"

#include <iostream>
#include <cmath>

/** Macro to easily define vector types. */
#define simd_vector(elcount, type)  __attribute__((vector_size((elcount)*sizeof(type)))) type

/** The last error */
int potential_err = potential_err_ok;


/** The null potential */
FPTYPE c_null[] = { FPTYPE_ZERO , FPTYPE_ZERO , FPTYPE_ZERO , FPTYPE_ZERO , FPTYPE_ZERO , FPTYPE_ZERO , FPTYPE_ZERO , FPTYPE_ZERO };
struct MxPotential potential_null = {
        PyObject_HEAD_INIT(&MxPotential_Type)
        NULL,
        {FPTYPE_ZERO , FPTYPE_ZERO , FPTYPE_ZERO , FPTYPE_ZERO } ,
        .c = c_null ,
        .a = 0.0 ,
        .b = DBL_MAX,
        POTENTIAL_NONE ,
        1
};


/* the error macro. */
#define error(id) ( potential_err = errs_register( id , potential_err_msg[-(id)] , __LINE__ , __FUNCTION__ , __FILE__ ) )

/* list of error messages. */
const char *potential_err_msg[] = {
		"Nothing bad happened.",
		"An unexpected NULL pointer was encountered.",
		"A call to malloc failed, probably due to insufficient memory.",
		"The requested value was out of bounds.",
		"Not yet implemented.",
		"Maximum number of intervals reached before tolerance satisfied."
};

static PyObject *potential_checkerr(MxPotential *p) {
    if(p == NULL) {
        std::string err = errs_getstring(0);
        PyErr_SetString(PyExc_ValueError, err.c_str());
    }
    return p;
}

static MxPotential *potential_alloc(PyTypeObject *type);

/**
 * @brief Switching function.
 *
 * @param r The radius.
 * @param A The start of the switching region.
 * @param B The end of the switching region.
 */

double potential_switch ( double r , double A , double B ) {

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

double potential_switch_p ( double r , double A , double B ) {

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

double potential_LJ126 ( double r , double A , double B ) {

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

double potential_LJ126_p ( double r , double A , double B ) {

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

double potential_LJ126_6p ( double r , double A , double B ) {

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

double potential_Coulomb ( double r ) {

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

double potential_Coulomb_p ( double r ) {

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

double potential_Coulomb_6p ( double r ) {

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

double potential_Ewald ( double r , double kappa ) {

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

double potential_Ewald_p ( double r , double kappa ) {

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

double potential_Ewald_6p ( double r , double kappa ) {

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

struct MxPotential *potential_create_harmonic ( double a , double b , double K , double r0 , double tol ) {

	struct MxPotential *p;

	/* allocate the potential */
	if ((p = potential_alloc(&MxPotential_Type)) == NULL ) {
		error(potential_err_malloc);
		return NULL;
	}

    p->flags = POTENTIAL_HARMONIC & POTENTIAL_R2 ;
    p->name = "Harmonic";

	/* fill this potential */
	potential_create_harmonic_K = K;
	potential_create_harmonic_r0 = r0;
	if ( potential_init(p,
                        &potential_create_harmonic_f,
                        &potential_create_harmonic_dfdr,
                        &potential_create_harmonic_d6fdr6,
                        a , b , tol ) < 0 ) {
		CAligned_Free(p);
		return NULL;
	}

	/* return it */
    return p;

}



double potential_create_linear_k;

/* the potential functions */
double potential_create_linear_f ( double r ) {
    return potential_create_linear_k * r;
}

double potential_create_linear_dfdr ( double r ) {
    return potential_create_linear_k;
}

double potential_create_linear_d6fdr6 ( double r ) {
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

struct MxPotential *potential_create_linear (double a , double b ,
                                             double k ,
                                             double tol ) {
    
    struct MxPotential *p;
    
    /* allocate the potential */
    if ((p = potential_alloc(&MxPotential_Type)) == NULL ) {
        error(potential_err_malloc);
        return NULL;
    }
    
    p->flags =  POTENTIAL_R2 ;
    p->name = "Linear";
    
    /* fill this potential */
    potential_create_linear_k = k;
    if ( potential_init( p , &potential_create_linear_f , NULL , &potential_create_linear_d6fdr6 , a , b , tol ) < 0 ) {
        CAligned_Free(p);
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

struct MxPotential *potential_create_harmonic_dihedral ( double K , int n , double delta , double tol ) {

	struct MxPotential *p;
	double a = -1.0, b = 1.0;

	/* Adjust end-points if delta is not a multiple of pi. */
	if ( fmod( delta , M_PI ) != 0 ) {
		a = -1.0 / (1.0 + sqrt(FPTYPE_EPSILON));
		b = 1.0 / (1.0 + sqrt(FPTYPE_EPSILON));
	}

	/* allocate the potential */
    if ((p = potential_alloc(&MxPotential_Type)) == NULL ) {
		error(potential_err_malloc);
		return NULL;
	}

    p->flags =   POTENTIAL_R | POTENTIAL_HARMONIC | POTENTIAL_DIHEDRAL;
    p->name = "Harmonic Dihedral";

	/* fill this potential */
	potential_create_harmonic_dihedral_K = K;
	potential_create_harmonic_dihedral_n = n;
	potential_create_harmonic_dihedral_delta = delta;
	if ( potential_init( p , &potential_create_harmonic_dihedral_f , NULL , &potential_create_harmonic_dihedral_d6fdr6 , a , b , tol ) < 0 ) {
		CAligned_Free(p);
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

struct MxPotential *potential_create_harmonic_angle ( double a , double b , double K , double theta0 , double tol ) {

	struct MxPotential *p;
	double left, right;

	/* allocate the potential */
    if ((p = potential_alloc(&MxPotential_Type)) == NULL ) {
		error(potential_err_malloc);
		return NULL;
	}

    p->flags = POTENTIAL_ANGLE | POTENTIAL_HARMONIC ;
    p->name = "Harmonic Angle";

	/* Adjust a and b accordingly. */
	if ( a < 0.0 )
		a = 0.0;
	if ( b > M_PI )
		b = M_PI;
	left = cos(b);
	right = cos(a);
	
    // the potential_init will automatically padd these already.
    //if ( left - fabs(left)*sqrt(FPTYPE_EPSILON) < -1.0 )
	//	left = -1.0 / ( 1.0 + sqrt(FPTYPE_EPSILON) );
	//if ( right + fabs(right)*sqrt(FPTYPE_EPSILON) > 1.0 )
	//	right = 1.0 / ( 1.0 + sqrt(FPTYPE_EPSILON) );

	/* fill this potential */
	potential_create_harmonic_angle_K = K;
	potential_create_harmonic_angle_theta0 = theta0;
	if ( potential_init( p , &potential_create_harmonic_angle_f , NULL , &potential_create_harmonic_angle_d6fdr6 , left , right , tol ) < 0 ) {
		CAligned_Free(p);
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

struct MxPotential *potential_create_Ewald ( double a , double b , double q , double kappa , double tol ) {

	struct MxPotential *p;

	/* allocate the potential */
    if ((p = potential_alloc(&MxPotential_Type)) == NULL ) {
		error(potential_err_malloc);
		return NULL;
	}

    p->flags =  POTENTIAL_R2 | POTENTIAL_EWALD ;
    p->name = "Ewald";

	/* fill this potential */
	potential_create_Ewald_q = q;
	potential_create_Ewald_kappa = kappa;
	if ( potential_init( p , &potential_create_Ewald_f , &potential_create_Ewald_dfdr , &potential_create_Ewald_d6fdr6 , a , b , tol ) < 0 ) {
		CAligned_Free(p);
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

struct MxPotential *potential_create_LJ126_Ewald ( double a , double b , double A , double B , double q , double kappa , double tol ) {

	struct MxPotential *p;

	/* allocate the potential */
    if ((p = potential_alloc(&MxPotential_Type)) == NULL ) {
		error(potential_err_malloc);
		return NULL;
	}

    p->flags =  POTENTIAL_R2 | POTENTIAL_LJ126 |  POTENTIAL_EWALD ;
    p->name = "Lennard-Jones Ewald";

	/* fill this potential */
	potential_create_LJ126_Ewald_A = A;
	potential_create_LJ126_Ewald_B = B;
	potential_create_LJ126_Ewald_kappa = kappa;
	potential_create_LJ126_Ewald_q = q;
	if ( potential_init( p , &potential_create_LJ126_Ewald_f , &potential_create_LJ126_Ewald_dfdr , &potential_create_LJ126_Ewald_d6fdr6 , a , b , tol ) < 0 ) {
		CAligned_Free(p);
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

struct MxPotential *potential_create_LJ126_Ewald_switch ( double a , double b , double A , double B , double q , double kappa , double s , double tol ) {

	struct MxPotential *p;

	/* allocate the potential */
    if ((p = potential_alloc(&MxPotential_Type)) == NULL ) {
		error(potential_err_malloc);
		return NULL;
	}

    p->flags =  POTENTIAL_R2 | POTENTIAL_LJ126 | POTENTIAL_EWALD | POTENTIAL_SWITCH ;
    p->name = "Lennard-Jones Ewald Switch";

	/* fill this potential */
	potential_create_LJ126_Ewald_switch_A = A;
	potential_create_LJ126_Ewald_switch_B = B;
	potential_create_LJ126_Ewald_switch_kappa = kappa;
	potential_create_LJ126_Ewald_switch_q = q;
	potential_create_LJ126_Ewald_switch_s = s;
	potential_create_LJ126_Ewald_switch_cutoff = b;
	if ( potential_init( p , &potential_create_LJ126_Ewald_switch_f , &potential_create_LJ126_Ewald_switch_dfdr , &potential_create_LJ126_Ewald_switch_d6fdr6 , a , b , tol ) < 0 ) {
		CAligned_Free(p);
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

struct MxPotential *potential_create_Coulomb ( double a , double b , double q , double tol ) {

	struct MxPotential *p;

	/* allocate the potential */
    if ((p = potential_alloc(&MxPotential_Type)) == NULL ) {
		error(potential_err_malloc);
		return NULL;
	}

    p->flags =  POTENTIAL_R2 |  POTENTIAL_COULOMB ;
    p->name = "Coulomb";

	/* fill this potential */
	potential_create_Coulomb_q = q;
	potential_create_Coulomb_b = b;
	if ( potential_init( p , &potential_create_Coulomb_f , &potential_create_Coulomb_dfdr , &potential_create_Coulomb_d6fdr6 , a , b , tol ) < 0 ) {
		CAligned_Free(p);
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

struct MxPotential *potential_create_LJ126_Coulomb ( double a , double b , double A , double B , double q , double tol ) {

	struct MxPotential *p;

	/* allocate the potential */
    if ((p = potential_alloc(&MxPotential_Type)) == NULL ) {
		error(potential_err_malloc);
		return NULL;
	}

    p->flags =  POTENTIAL_R2 | POTENTIAL_COULOMB | POTENTIAL_LJ126  ;
    p->name = "Lennard-Jones Coulomb";

	/* fill this potential */
	potential_create_LJ126_Coulomb_q = q;
	potential_create_LJ126_Coulomb_b = b;
	potential_create_LJ126_Coulomb_A = A;
	potential_create_LJ126_Coulomb_B = B;
	if ( potential_init( p , &potential_create_LJ126_Coulomb_f , &potential_create_LJ126_Coulomb_dfdr , &potential_create_LJ126_Coulomb_d6fdr6 , a , b , tol ) < 0 ) {
		CAligned_Free(p);
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

struct MxPotential *potential_create_LJ126 ( double a , double b , double A , double B , double tol ) {

    MxPotential *p = NULL;

	/* allocate the potential */
    if ((p = potential_alloc(&MxPotential_Type)) == NULL ) {
		error(potential_err_malloc);
		return NULL;
 	}

    p->flags =  POTENTIAL_R2  | POTENTIAL_LJ126 ;
    p->name = "Lennard-Jones";

	/* fill this potential */
	potential_create_LJ126_A = A;
	potential_create_LJ126_B = B;
	if ( potential_init( p , &potential_create_LJ126_f , &potential_create_LJ126_dfdr , &potential_create_LJ126_d6fdr6 , a , b , tol ) < 0 ) {
		CAligned_Free(p);
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

struct MxPotential *potential_create_LJ126_switch ( double a , double b , double A , double B , double s , double tol ) {

	struct MxPotential *p;

	/* allocate the potential */
    if ((p = potential_alloc(&MxPotential_Type)) == NULL ) {
		error(potential_err_malloc);
		return NULL;
	}

    p->flags =  POTENTIAL_R2 | POTENTIAL_LJ126 | POTENTIAL_SWITCH ;
    p->name = "Lennard-Jones Switch";

	/* fill this potential */
	potential_create_LJ126_switch_A = A;
	potential_create_LJ126_switch_B = B;
	potential_create_LJ126_switch_s = s;
	potential_create_LJ126_switch_cutoff = b;
	if ( potential_init( p , &potential_create_LJ126_switch_f , &potential_create_LJ126_switch_dfdr , &potential_create_LJ126_switch_d6fdr6 , a , b , tol ) < 0 ) {
		CAligned_Free(p);
		return NULL;
	}

	/* return it */
			return p;

}



#define Power(base, exp) std::pow(base, exp)

#define Log(x) std::log(x)

static double potential_create_SS_e;
static double potential_create_SS_k;
static double potential_create_SS_r0;
static double potential_create_SS_v0_r;

static double potential_create_SS_linear_f(double eta, double r) {
    return potential_create_SS_k - (Power(2,1/eta)*potential_create_SS_k*r)/potential_create_SS_r0;
}

static double potential_create_SS_linear_dfdr(double eta) {
    return -((Power(2,1/eta)*potential_create_SS_k)/potential_create_SS_r0);
}

/* the potential functions */
// {Solve[ff == 0, r][[1]], ff, D[ff, {r, 1}], D[ff, {r, 6}]} /. \[Eta] -> 1 // Simplify
// {{r -> r0/2}, (e r0 (-2 r + r0))/r^2, (2 e (r - r0) r0)/r^3, -((720 e (2 r - 7 r0) r0)/r^8)}
// List(List(Rule(r,r0/2.)),(e*r0*(-2*r + r0))/Power(r,2),(2*e*(r - r0)*r0)/Power(r,3),(-720*e*(2*r - 7*r0)*r0)/Power(r,8))

static double potential_create_SS1_f ( double r ) {
    double e =    potential_create_SS_e;
    double r0 =   potential_create_SS_r0;
    double v0_r = potential_create_SS_v0_r;
    
    if(r < v0_r) {
        return potential_create_SS_linear_f(1, r);
    }
    else {
        return (e*r0*(-2*r + r0))/Power(r,2);
    }
}

static double potential_create_SS1_dfdr ( double r ) {
    double e =    potential_create_SS_e;
    double r0 =   potential_create_SS_r0;
    double v0_r = potential_create_SS_v0_r;
    
    if(r < v0_r) {
        return potential_create_SS_linear_dfdr(1);
    }
    else {
        return (2*e*(r - r0)*r0)/Power(r,3);
    }
}

static double potential_create_SS1_d6fdr6 ( double r ) {
    double e =    potential_create_SS_e;
    double r0 =   potential_create_SS_r0;
    double v0_r = potential_create_SS_v0_r;
    
    if(r < v0_r) {
        return 0;
    }
    else {
        return (-720*e*(2*r - 7*r0)*r0)/Power(r,8);
    }
}

struct MxPotential *potential_create_SS1(double k, double e, double r0, double a , double b ,double tol) {
    
    struct MxPotential *p;
    
    /* allocate the potential */
    if ((p = potential_alloc(&MxPotential_Type)) == NULL ) {
        error(potential_err_malloc);
        return NULL;
    }
    
    p->flags =  POTENTIAL_R2 | POTENTIAL_LJ126 | POTENTIAL_SWITCH ;
    p->name = "Soft Sphere 1";
    
    potential_create_SS_e = e;
    potential_create_SS_k = k;
    potential_create_SS_r0 = r0;
    potential_create_SS_v0_r = r0/2.;
    
    int err = 0;
    
    if((err = potential_init(p ,&potential_create_SS1_f,
        &potential_create_SS1_dfdr , &potential_create_SS1_d6fdr6 , a , b , tol )) < 0 ) {
        
        std::cout << "error creating potential: " << potential_err_msg[-err] << std::endl;
		CAligned_Free(p);
        return NULL;
    }
    
    /* return it */
    return p;
}


/* the potential functions */
// {Solve[ff == 0, r][[1]], ff, D[ff, {r, 1}], D[ff, {r, 6}]} /. \[Eta] -> 2 // Simplify
// {{r -> r0/Sqrt[2]},
// (e r0^2 (-2 r^2 + r0^2))/r^4,
// (4 e r0^2 (r^2 - r0^2))/r^5,
// -((10080 e r0^2 (r^2 - 6 r0^2))/r^10)}
// Rule(r,r0/Sqrt(2)),
// (e*Power(r0,2)*(-2*Power(r,2) + Power(r0,2)))/Power(r,4),
// (4*e*Power(r0,2)*(Power(r,2) - Power(r0,2)))/Power(r,5),
// (-10080*e*Power(r0,2)*(Power(r,2) - 6*Power(r0,2)))/Power(r,10))

static double potential_create_SS2_f ( double r ) {
    double e =    potential_create_SS_e;
    double r0 =   potential_create_SS_r0;
    double v0_r = potential_create_SS_v0_r;
    
    if(r < v0_r) {
        return potential_create_SS_linear_f(2, r);
    }
    else {
        return (e*Power(r0,2)*(-2*Power(r,2) + Power(r0,2)))/Power(r,4);
    }
}

static double potential_create_SS2_dfdr ( double r ) {
    double e =    potential_create_SS_e;
    double r0 =   potential_create_SS_r0;
    double v0_r = potential_create_SS_v0_r;
    
    if(r < v0_r) {
        return potential_create_SS_linear_dfdr(2);
    }
    else {
        return (4*e*Power(r0,2)*(Power(r,2) - Power(r0,2)))/Power(r,5);
    }
}

static double potential_create_SS2_d6fdr6 ( double r ) {
    double e =    potential_create_SS_e;
    double r0 =   potential_create_SS_r0;
    double v0_r = potential_create_SS_v0_r;
    
    if(r < v0_r) {
        return 0;
    }
    else {
        // -((10080 e r0^2 (r^2 - 6 r0^2))/r^10)}
        return (-10080*e*Power(r0,2)*(Power(r,2) - 6*Power(r0,2)))/Power(r,10);
    }
}

struct MxPotential *potential_create_SS2(double k, double e, double r0, double a , double b ,double tol) {
    
    struct MxPotential *p;
    
    /* allocate the potential */
    if ((p = potential_alloc(&MxPotential_Type)) == NULL ) {
        error(potential_err_malloc);
        return NULL;
    }
    
    p->flags =  POTENTIAL_R2 | POTENTIAL_LJ126 | POTENTIAL_SWITCH ;
    p->name = "Soft Sphere 2";
    
    potential_create_SS_e = e;
    potential_create_SS_k = k;
    potential_create_SS_r0 = r0;
    potential_create_SS_v0_r = r0/std::sqrt(2.);
    
    int err = 0;
    
    if((err = potential_init(p ,&potential_create_SS2_f,
                             &potential_create_SS2_dfdr ,
                             &potential_create_SS2_d6fdr6 , a , b , tol )) < 0 ) {
        
        std::cout << "error creating potential: " << potential_err_msg[-err] << std::endl;
		CAligned_Free(p);
        return NULL;
    }
    
    /* return it */
    return p;
}

/* the potential functions */
// {Solve[ff == 0, r][[1]], ff, D[ff, {r, 1}],D[ff, {r, 6}]} /. \[Eta] -> 3 // Simplify
// {{r -> r0/2^(1/3)},
// (e r0^3 (-2 r^3 + r0^3))/r^6,
// (6 e r0^3 (r^3 - r0^3))/r^7,
// -((10080 e (4 r^3 r0^3 - 33 r0^6))/r^12)}
// Rule(r,r0/Power(2,0.3333333333333333)),
// (e*Power(r0,3)*(-2*Power(r,3) + Power(r0,3)))/Power(r,6),
// (6*e*Power(r0,3)*(Power(r,3) - Power(r0,3)))/Power(r,7),
// (-10080*e*(4*Power(r,3)*Power(r0,3) - 33*Power(r0,6)))/Power(r,12)


static double potential_create_SS3_f ( double r ) {
    double e =    potential_create_SS_e;
    double r0 =   potential_create_SS_r0;
    double v0_r = potential_create_SS_v0_r;
    
    if(r < v0_r) {
        return potential_create_SS_linear_f(3, r);
    }
    else {
        return (e*Power(r0,3)*(-2*Power(r,3) + Power(r0,3)))/Power(r,6);
    }
}

static double potential_create_SS3_dfdr ( double r ) {
    double e =    potential_create_SS_e;
    double r0 =   potential_create_SS_r0;
    double v0_r = potential_create_SS_v0_r;
    
    if(r < v0_r) {
        return potential_create_SS_linear_dfdr(3);
    }
    else {
        return (6*e*Power(r0,3)*(Power(r,3) - Power(r0,3)))/Power(r,7);
    }
}

static double potential_create_SS3_d6fdr6 ( double r ) {
    double e =    potential_create_SS_e;
    double r0 =   potential_create_SS_r0;
    double v0_r = potential_create_SS_v0_r;
    
    if(r < v0_r) {
        return 0;
    }
    else {
        // -((10080 e (4 r^3 r0^3 - 33 r0^6))/r^12)
        return (-10080*e*(4*Power(r,3)*Power(r0,3) - 33*Power(r0,6)))/Power(r,12);
    }
}

struct MxPotential *potential_create_SS3(double k, double e, double r0, double a , double b ,double tol) {
    
    struct MxPotential *p;
    
    /* allocate the potential */
    if ((p = potential_alloc(&MxPotential_Type)) == NULL ) {
        error(potential_err_malloc);
        return NULL;
    }
    
    p->flags =  POTENTIAL_R2 | POTENTIAL_LJ126 | POTENTIAL_SWITCH ;
    p->name = "Soft Sphere 3";
    
    potential_create_SS_e = e;
    potential_create_SS_k = k;
    potential_create_SS_r0 = r0;
    potential_create_SS_v0_r = r0/Power(2,0.3333333333333333);
    
    int err = 0;
    
    if((err = potential_init(p ,&potential_create_SS3_f,
                             &potential_create_SS3_dfdr ,
                             &potential_create_SS3_d6fdr6 , a , b , tol )) < 0 ) {
        
        std::cout << "error creating potential: " << potential_err_msg[-err] << std::endl;
		CAligned_Free(p);
        return NULL;
    }
    
    /* return it */
    return p;
}



/* the potential functions */
// {Solve[ff == 0, r][[1]], ff, D[ff, {r, 1}], D[ff, {r, 6}]} /. \[Eta] -> 4 // Simplify
// {r -> r0/2^(1/4)},
// (e r0^4 (-2 r^4 + r0^4))/r^8,
// (8 e r0^4 (r^4 - r0^4))/r^9, -((8640 e (14 r^4 r0^4 - 143 r0^8))/r^14)
// List(Rule(r,r0/Power(2,0.25))),
// (e*Power(r0,4)*(-2*Power(r,4) + Power(r0,4)))/Power(r,8),
// (8*e*Power(r0,4)*(Power(r,4) - Power(r0,4)))/Power(r,9),
// (-8640*e*(14*Power(r,4)*Power(r0,4) - 143*Power(r0,8)))/Power(r,14)


static double potential_create_SS4_f ( double r ) {
    double e =    potential_create_SS_e;
    double r0 =   potential_create_SS_r0;
    double v0_r = potential_create_SS_v0_r;
    
    if(r < v0_r) {
        return potential_create_SS_linear_f(4, r);
    }
    else {
        return (e*Power(r0,4)*(-2*Power(r,4) + Power(r0,4)))/Power(r,8);
    }
}

static double potential_create_SS4_dfdr ( double r ) {
    double e =    potential_create_SS_e;
    double r0 =   potential_create_SS_r0;
    double v0_r = potential_create_SS_v0_r;
    
    if(r < v0_r) {
        return potential_create_SS_linear_dfdr(4);
    }
    else {
        return (8*e*Power(r0,4)*(Power(r,4) - Power(r0,4)))/Power(r,9);
    }
}

static double potential_create_SS4_d6fdr6 ( double r ) {
    double e =    potential_create_SS_e;
    double r0 =   potential_create_SS_r0;
    double v0_r = potential_create_SS_v0_r;
    
    if(r < v0_r) {
        return 0;
    }
    else {
        // (8 e r0^4 (r^4 - r0^4))/r^9, -((8640 e (14 r^4 r0^4 - 143 r0^8))/r^14)
        return (-8640*e*(14*Power(r,4)*Power(r0,4) - 143*Power(r0,8)))/Power(r,14);
    }
}

struct MxPotential *potential_create_SS4(double k, double e, double r0, double a , double b ,double tol) {
    
    struct MxPotential *p;
    
    /* allocate the potential */
    if ((p = potential_alloc(&MxPotential_Type)) == NULL ) {
        error(potential_err_malloc);
        return NULL;
    }
    
    p->flags =  POTENTIAL_R2 | POTENTIAL_LJ126 | POTENTIAL_SWITCH ;
    p->name = "Soft Sphere 4";
    
    potential_create_SS_e = e;
    potential_create_SS_k = k;
    potential_create_SS_r0 = r0;
    potential_create_SS_v0_r = r0/Power(2,0.25);
    
    int err = 0;
    
    if((err = potential_init(p ,&potential_create_SS4_f,
                             &potential_create_SS4_dfdr ,
                             &potential_create_SS4_d6fdr6 , a , b , tol )) < 0 ) {
        
        std::cout << "error creating potential: " << potential_err_msg[-err] << std::endl;
		CAligned_Free(p);
        return NULL;
    }
    
    /* return it */
    return p;
}


struct MxPotential *potential_create_SS(int eta, double k, double e, double r0,
                                        double a , double b , double tol, bool shift) {
    
    MxPotential *result = NULL;
    
    if(eta == 1) {
        result =  potential_create_SS1(k, e, r0, a, b, tol);
    }
    else if(eta == 2) {
        result =  potential_create_SS2(k, e, r0, a, b, tol);
    }
    else if(eta == 3) {
        result = potential_create_SS3(k, e, r0, a, b, tol);
    }
    else if(eta == 4) {
        result = potential_create_SS4(k, e, r0, a, b, tol);
    }
    
    if(result && shift) {
        result->flags |= POTENTIAL_SHIFTED;
        result->r0 = r0;
    }
    
    return result;
}

/**
 * @brief Free the memory associated with the given potential.
 * 
 * @param p Pointer to the #potential to clear.
 */

void potential_clear ( struct MxPotential *p ) {

	/* Do nothing? */
	if ( p == NULL )
		return;

	/* Clear the flags. */
	p->flags = POTENTIAL_NONE;

	/* Clear the coefficients. */
	CAligned_Free( p->c );
	p->c = NULL;

}


static double overlapping_sphere_k;
static double overlapping_sphere_mu;
static double overlapping_sphere_k_harmonic;
static double overlapping_sphere_harmonic_r0;
static double overlapping_sphere_harmonic_k;

// overlapping sphere f
//Piecewise(List(List(-x + x*Log(x),x <= 1)),-1 + Power(k,-2) - (Power(E,k - k*x)*(1 + k*(-1 + x)))/Power(k,2)),
static double overlapping_sphere_f ( double x ) {
    double k = overlapping_sphere_k;
    double mu = overlapping_sphere_mu;
    double harmonic_r0 = overlapping_sphere_harmonic_r0;
    double harmonic_k = overlapping_sphere_harmonic_k;
    
    double result;
    if(x <= 1) {
        result =  -x + x*Log(x);
    }
    else {
        result =  -1 + Power(k,-2) - (Power(M_E,k - k*x)*(1 + k*(-1 + x)))/Power(k,2);
    }
    
    //std::cout << "fdata = Append[fdata, {" << x << ", " << result << "}];" << std::endl;
    return mu * result + harmonic_k*Power(-x + harmonic_r0,2);
    //return harmonic_k*Power(-x + harmonic_r0,2);
}

// overlapping sphere fp
//Piecewise(List(List(Log(x),x < 1),List(Power(E,k*(1 - x))*(-1 + x),x >= 1)),0),
static double overlapping_sphere_fp ( double x ) {
    double k = overlapping_sphere_k;
    double mu = overlapping_sphere_mu;
    double harmonic_r0 = overlapping_sphere_harmonic_r0;
    double harmonic_k = overlapping_sphere_harmonic_k;
    double result;
    if(x <= 1) {
        result = Log(x);
    }
    else {
        result =  Power(M_E,k*(1 - x))*(-1 + x);
    }
    
    //std::cout << "fpdata = Append[fpdata, {" << x << ", " << result << "}];" << std::endl;
    return  mu * result + 2*harmonic_k*(-x + harmonic_r0);
    //return 2*harmonic_k*(-x + harmonic_r0);
}

// overlapping sphere f6p
//Piecewise(List(List(24/Power(x,5),x < 1),List(Power(E,k - k*x)*Power(k,4) -
//    Power(E,k - k*x)*Power(k,4)*(-4 - k + k*x),x > 1)),Indeterminate)
static double overlapping_sphere_f6p ( double x ) {
    
    double k = overlapping_sphere_k;
    double mu = overlapping_sphere_mu;
    double result;
    if(x <= 1) {
        result =  24/Power(x,5);
    }
    else {
        result =  Power(M_E,k - k*x)*Power(k,4) - Power(M_E,k - k*x)*Power(k,4)*(-4 - k + k*x);
    }
    //std::cout << "fp6data = Append[fp6data, {" << x << ", " << result << "}];" << std::endl;
    return mu * result;
    //return 0;
}

struct MxPotential *potential_create_overlapping_sphere(double mu, double k,
    double harmonic_k, double harmonic_r0,
    double a , double b ,double tol) {
    
    struct MxPotential *p;
    
    /* allocate the potential */
    if ((p = potential_alloc(&MxPotential_Type)) == NULL ) {
        error(potential_err_malloc);
        return NULL;
    }
    
    overlapping_sphere_mu = mu;
    overlapping_sphere_k = k;
    overlapping_sphere_harmonic_k = harmonic_k;
    overlapping_sphere_harmonic_r0 = harmonic_r0;
    
    
    p->flags =  POTENTIAL_R2  | POTENTIAL_LJ126 | POTENTIAL_SCALED;
    
    if(harmonic_k == 0.0) {
        p->name = "Overlapping Sphere";
    }
    else {
        p->name = "Overlapping Sphere with Harmonic";
    }
    
    int err = 0;
    
    if((err = potential_init(p ,&overlapping_sphere_f,
                             &overlapping_sphere_fp ,
                             &overlapping_sphere_f6p , a , b , tol )) < 0 ) {
        
        std::cout << "error creating potential: " << potential_err_msg[-err] << std::endl;
        CAligned_Free(p);
        return NULL;
    }
    
    /* return it */
    return p;
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

int potential_init (struct MxPotential *p ,
                    double (*f)( double ) ,
                    double (*fp)( double ) ,
                    double (*f6p)( double ) ,
                    FPTYPE a , FPTYPE b , FPTYPE tol ) {

	double alpha, w;
	int l = potential_ivalsa, r = potential_ivalsb, m;
	FPTYPE err_l = 0, err_r = 0, err_m = 0;
	FPTYPE *xi_l = NULL, *xi_r = NULL, *xi_m = NULL;
	FPTYPE *c_l = NULL, *c_r = NULL, *c_m = NULL;
	int i = 0, k = 0;
	double e;
	FPTYPE mtol = 10 * FPTYPE_EPSILON;

	/* check inputs */
	if ( p == NULL || f == NULL )
		return error(potential_err_null);

	/* check if we have a user-specified 6th derivative or not. */
	if ( f6p == NULL )
		return error(potential_err_nyi);
    
    /* set the boundaries */
    p->a = a; p->b = b;

	/* Stretch the domain ever so slightly to accommodate for rounding
       error when computing the index. */
	b += fabs(b) * sqrt(FPTYPE_EPSILON);
	a -= fabs(a) * sqrt(FPTYPE_EPSILON);
	// printf( "potential_init: setting a=%.16e, b=%.16e.\n" , a , b );

	

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
	xi_l = (FPTYPE *)CAligned_Malloc( sizeof(FPTYPE) * (l + 1), potential_align );
	c_l = (FPTYPE *)CAligned_Malloc( sizeof(FPTYPE) * (l+1) * potential_chunk, potential_align);
	if (xi_l == NULL || c_l == NULL) {
		return error(potential_err_malloc);
    }
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
		CAligned_Free(xi_l);
        
        // double test_n = p->alpha[0] + p->b * (p->alpha[1] + p->b * p->alpha[2]);
        
        //int ttn = test_n;
        
        //int tttn = int(FPTYPE_FMAX( FPTYPE_ZERO , p->alpha[0] + p->b * (p->alpha[1] + p->b * p->alpha[2])));
        
        
        assert(int(FPTYPE_FMAX( FPTYPE_ZERO , p->alpha[0] + p->b * (p->alpha[1] + p->b * p->alpha[2]))) < p->n + 1);
		return potential_err_ok;
	}

	/* loop until we have an upper bound on the right... */
	while ( 1 ) {

		/* compute the larger interpolation... */
		/* printf("potential_init: trying r=%i...\n",r); fflush(stdout); */
		xi_r = (FPTYPE*)CAligned_Malloc(sizeof(FPTYPE) * (r + 1),  potential_align );
		c_r =  (FPTYPE*)CAligned_Malloc(sizeof(FPTYPE) * (r + 1) * potential_chunk, potential_align);
        if ( xi_r == NULL || c_r == NULL) {
			return error(potential_err_malloc);
        }
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
        if ( potential_getcoeffs(f,fp,xi_r,r,&c_r[potential_chunk],&err_r) < 0 ) {
			return error(potential_err);
        }
		/* printf("potential_init: err_r=%22.16e.\n",err_r); fflush(stdout); */

		/* if this is better than tolerance, break... */
        if ( err_r < tol ) {
			break;
        }

		/* Have we too many intervals? */
		else if ( 2*r > potential_ivalsmax ) {
			/* printf( "potential_init: warning: maximum nr of intervals (%i) reached, err=%e.\n" , r , err_r );
            break; */
			return error(potential_err_ivalsmax);
		}

		/* otherwise, l=r and r = 2*r */
		else {
			l = r; err_l = err_r;
			CAligned_Free(xi_l); xi_l = xi_r;
			CAligned_Free(c_l); c_l = c_r;
			r *= 2;
		}

	} /* loop until we have a good right estimate */

	/* we now have a left and right estimate -- binary search! */
	while ( r - l > 1 ) {

		/* find the middle */
		m = 0.5 * ( r + l );

		/* construct that interpolation */
		/* printf("potential_init: trying m=%i...\n",m); fflush(stdout); */
		xi_m = (FPTYPE*)CAligned_Malloc(sizeof(FPTYPE) * (m + 1), potential_align);
		c_m =  (FPTYPE*)CAligned_Malloc(sizeof(FPTYPE) * (m + 1) * potential_chunk, potential_align);

        if ( xi_m == NULL || c_m == NULL ) {
			return error(potential_err_malloc);
        }
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
					CAligned_Free(xi_l); xi_l = xi_m;
					CAligned_Free(c_l); c_l = c_m;
				}

				/* otherwise, go right... */
				else {
					r = m; err_r = err_m;
					CAligned_Free(xi_r); xi_r = xi_m;
					CAligned_Free(c_r); c_r = c_m;
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
	CAligned_Free(xi_r);
	CAligned_Free(xi_l);
	CAligned_Free(c_l);

	/* all is well that ends well... */
    
    // round off error sometimes lets max r larger than number of bins,
    // the potential eval takes care of not overstepping bin count.
    //float max_n = p->alpha[0] + p->b * (p->alpha[1] + p->b * p->alpha[2]);
    //assert(FPTYPE_FMAX( FPTYPE_ZERO , max_n < p->n+1));
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

#include <pybind11/pybind11.h>
namespace py = pybind11;

static MxPotential *potential_alloc(PyTypeObject *type) {

    std::cout << MX_FUNCTION << std::endl;

    struct MxPotential *obj = NULL;

    /* allocate the potential */
    if ((obj = (MxPotential * )CAligned_Malloc(type->tp_basicsize, 16 )) == NULL ) {
        return NULL;
    }
    
    ::memset(obj, NULL, type->tp_basicsize);

    if (type->tp_flags & Py_TPFLAGS_HEAPTYPE)
        Py_INCREF(type);

    PyObject_INIT(obj, type);

    if (PyType_IS_GC(type)) {
        assert(0 && "should not get here");
        //  _PyObject_GC_TRACK(obj);
    }

    return obj;
}

static void potential_dealloc(PyObject* obj) {
	std::cout << MX_FUNCTION << std::endl;
	CAligned_Free(obj);
}

static PyObject *potential_call(PyObject *_self, PyObject *_args, PyObject *_kwargs) {
    MxPotential *self = (MxPotential*)_self;

    try {
        py::args args = py::reinterpret_borrow<py::args>(_args);
        py::kwargs kwargs = py::reinterpret_borrow<py::kwargs>(_kwargs);
        

        float r = py::cast<float>(args[0]);
        
        
        double ri = arg<double>("ri",  1, _args, _kwargs, -1);
        double rj = arg<double>("rj",  2, _args, _kwargs, -1);
        
        // if no r args are given, we pull the r0 from the potential,
        // and use the ri, rj to cancel them out.
        if((self->flags & POTENTIAL_SHIFTED) && ri < 0 && rj < 0) {
            ri = self->r0 / 2;
            rj = self->r0 / 2;
        }
        
        // if no r args are given, we pull the r0 from the potential,
        // and use the ri, rj to cancel them out.
        if((self->flags & POTENTIAL_SCALED)) {
            double s = arg<double>("s",  1, _args, _kwargs, -1);
            if(s < 0) {
                PyErr_Warn(PyExc_Warning, "calling scaled potential without s, sum of particle radii");
                ri = 1 / 2;
                rj = 1 / 2;
            }
            else {
                ri = rj = s / 2;
            }
        }
        
        float e = 0;
        float f = 0;
        
        if(self->flags & POTENTIAL_R) {
            potential_eval_r(self, r, &e, &f);
        }
        else {
            potential_eval_ex(self, ri, rj, r*r, &e, &f);
        }
        
        f = f * r;
        
        PyObject *res = PyTuple_New(2);
        PyTuple_SET_ITEM(res, 0, PyFloat_FromDouble(e));
        PyTuple_SET_ITEM(res, 1, PyFloat_FromDouble(f));
        
        return res;
    }
    catch (const pybind11::builtin_exception &e) {
        e.set_error();
        return NULL;
    }
    catch(py::error_already_set &e){
        e.restore();
        return NULL;
    }
}

static PyObject *potential_force(PyObject *_self, PyObject *_args, PyObject *_kwargs) {
    MxPotential *self = (MxPotential*)_self;
    
    try {
        py::args args = py::reinterpret_borrow<py::args>(_args);
        py::kwargs kwargs = py::reinterpret_borrow<py::kwargs>(_kwargs);
        
        
        float r = py::cast<float>(args[0]);
        
        
        double ri = arg<double>("ri",  1, _args, _kwargs, -1);
        double rj = arg<double>("rj",  2, _args, _kwargs, -1);
        
        // if no r args are given, we pull the r0 from the potential,
        // and use the ri, rj to cancel them out.
        if((self->flags & POTENTIAL_SHIFTED) && ri < 0 && rj < 0) {
            ri = self->r0 / 2;
            rj = self->r0 / 2;
        }
        
        float e = 0;
        float f = 0;
        
        if(self->flags & POTENTIAL_R) {
            potential_eval_r(self, r, &e, &f);
        }
        else {
            potential_eval_ex(self, ri, rj, r*r, &e, &f);
        }
        
     //   if (potential_eval_ex(pot, part_i->radius, part_j->radius, r2 , &e , &f )) {
     //
     //       for ( k = 0 ; k < 3 ; k++ ) {
     //           w = f * dx[k];
     //           pif[k] -= w;
     //           part_j->f[k] += w;
     //       }
        
        f = (f * r) / 2;
        
        return py::cast(f).release().ptr();
    }
    catch (const pybind11::builtin_exception &e) {
        e.set_error();
        return NULL;
    }
    catch(py::error_already_set &e){
        e.restore();
        return NULL;
    }
}

static PyObject *_lennard_jones_12_6(PyObject *_self, PyObject *_args, PyObject *_kwargs) {
    std::cout << MX_FUNCTION << std::endl;
    
    try {
        double min = arg<double>("min", 0, _args, _kwargs);
        double max = arg<double>("max", 1, _args, _kwargs);
        double A = arg<double>("A", 2, _args, _kwargs);
        double B = arg<double>("B", 3, _args, _kwargs);
        double tol = arg<double>("tol", 4, _args, _kwargs, 0.001 * (max-min));
        return potential_create_LJ126( min, max, A, B, tol);
    }
    catch (const std::exception &e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        return NULL;
    }
    catch(py::error_already_set &e){
        e.restore();
        return NULL;
    }
}


static PyObject *_lennard_jones_12_6_coulomb(PyObject *_self, PyObject *_args, PyObject *_kwargs) {
    std::cout << MX_FUNCTION << std::endl;

    try {
        double min = arg<double>("min", 0, _args, _kwargs);
        double max = arg<double>("max", 1, _args, _kwargs);
        double A = arg<double>("A", 2, _args, _kwargs);
        double B = arg<double>("B", 3, _args, _kwargs);
        double q = arg<double>("q", 4, _args, _kwargs);
        double tol = arg<double>("tol", 5, _args, _kwargs, 0.001 * (max-min));
        return potential_checkerr(potential_create_LJ126_Coulomb( min, max, A, B, q, tol));
    }
    catch (const std::exception &e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        return NULL;
    }
    catch(py::error_already_set &e){
        e.restore();
        return NULL;
    }
}

static PyObject *_soft_sphere(PyObject *_self, PyObject *_args, PyObject *_kwargs) {
    std::cout << MX_FUNCTION << std::endl;

    try {
        double kappa = arg<double>("kappa", 0, _args, _kwargs);
        double epsilon = arg<double>("epsilon", 1, _args, _kwargs);
        double r0 = arg<double>("r0", 2, _args, _kwargs);
        double eta = arg<double>("eta", 3, _args, _kwargs);
        double min = arg<double>("min", 4, _args, _kwargs, 0);
        double max = arg<double>("max", 5, _args, _kwargs, 2);
        double tol = arg<double>("tol", 6, _args, _kwargs, 0.001 * (max-min));
        bool shift = arg<bool>("shift", 7, _args, _kwargs, false);
        return potential_checkerr(potential_create_SS(eta, kappa, epsilon, r0, min, max, tol, shift));
    }
    catch (const std::exception &e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        return NULL;
    }
    catch(py::error_already_set &e){
        e.restore();
        return NULL;
    }
}

static PyObject *_ewald(PyObject *_self, PyObject *_args, PyObject *_kwargs) {
    std::cout << MX_FUNCTION << std::endl;

    try {
        double min = arg<double>("min", 0, _args, _kwargs);
        double max = arg<double>("max", 1, _args, _kwargs);
        double q = arg<double>("q", 2, _args, _kwargs);
        double kappa = arg<double>("kappa", 3, _args, _kwargs);
        double tol = arg<double>("tol", 4, _args, _kwargs, 0.001 * (max-min));
        return potential_checkerr(potential_create_Ewald( min, max, q, kappa, tol));
    }
    catch (const std::exception &e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        return NULL;
    }
    catch(py::error_already_set &e){
        e.restore();
        return NULL;
    }
}


static PyObject *_coulomb(PyObject *_self, PyObject *_args, PyObject *_kwargs) {
    std::cout << MX_FUNCTION << std::endl;
    
    try {
        double q = arg<double>("q", 0, _args, _kwargs);
        double min = arg<double>("min", 1, _args, _kwargs, 0.01);
        double max = arg<double>("max", 2, _args, _kwargs, 2);
        double tol = arg<double>("tol", 3, _args, _kwargs, 0.01 * (max-min));
        return potential_checkerr(potential_create_Coulomb( min, max, q, tol));
    }
    catch (const std::exception &e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        return NULL;
    }
    catch(py::error_already_set &e){
        e.restore();
        return NULL;
    }
}


static PyObject *_harmonic(PyObject *_self, PyObject *_args, PyObject *_kwargs){
    std::cout << MX_FUNCTION << std::endl;

    try {
        double k =     arg<double>("k", 0, _args, _kwargs);
        double r0 =    arg<double>("r0", 1, _args, _kwargs);
        double range = r0;
        double min =   arg<double>("min", 2, _args, _kwargs, r0 - range);
        double max =   arg<double>("max", 3, _args, _kwargs, r0 + range);
        double tol =   arg<double>("tol", 4, _args, _kwargs, 0.01 * (max-min));
        return potential_checkerr(potential_create_harmonic(min, max, k, r0, tol));
    }
    catch (const std::exception &e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        return NULL;
    }
    catch(py::error_already_set &e){
        e.restore();
        return NULL;
    }
}

static PyObject *_linear(PyObject *_self, PyObject *_args, PyObject *_kwargs){
    std::cout << MX_FUNCTION << std::endl;
    
    try {
        double k =     arg<double>("k", 0, _args, _kwargs);
        double min =   arg<double>("min", 1, _args, _kwargs, std::numeric_limits<double>::epsilon());
        double max =   arg<double>("max", 2, _args, _kwargs, 10);
        double tol =   arg<double>("tol", 3, _args, _kwargs, 0.01 * (max-min));
        return potential_checkerr(potential_create_linear(min, max, k, tol));
    }
    catch (const std::exception &e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        return NULL;
    }
    catch(py::error_already_set &e){
        e.restore();
        return NULL;
    }
}

static PyObject *_harmonic_angle(PyObject *_self, PyObject *_args, PyObject *_kwargs) {
    std::cout << MX_FUNCTION << std::endl;

    try {
        double k = arg<double>("k", 0, _args, _kwargs);
        double theta0 = arg<double>("theta0", 1, _args, _kwargs);
        double min = arg<double>("min", 2, _args, _kwargs, 0.0);
        double max = arg<double>("max", 3, _args, _kwargs, M_PI);
        double tol = arg<double>("tol", 4, _args, _kwargs, 0.005 * std::abs(max-min));

        return potential_checkerr(potential_create_harmonic_angle( min, max, k, theta0, tol));
    }
    catch (const std::exception &e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        return NULL;
    }
    catch(py::error_already_set &e){
        e.restore();
        return NULL;
    }
}


static PyObject *_harmonic_dihedral(PyObject *_self, PyObject *_args, PyObject *_kwargs) {
    std::cout << MX_FUNCTION << std::endl;
    
    try {
        double k = arg<double>("k", 0, _args, _kwargs);
        int n = arg<int>("n", 1, _args, _kwargs);
        double delta = arg<double>("delta", 2, _args, _kwargs);
        double tol = arg<double>("tol", 3, _args, _kwargs, 0.001);
        return potential_checkerr(potential_create_harmonic_dihedral( k, n, delta, tol));
    }
    catch (const std::exception &e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        return NULL;
    }
    catch(py::error_already_set &e){
        e.restore();
        return NULL;
    }
}

// potential_create_well(double k, double n, double r0, double tol, double min, double max)

static PyObject *_well(PyObject *_self, PyObject *_args, PyObject *_kwargs) {
    std::cout << MX_FUNCTION << std::endl;

    try {
        double k =   arg<double>("k",   0, _args, _kwargs);
        double n =   arg<double>("n",   1, _args, _kwargs);
        double r0 =  arg<double>("r0",  2, _args, _kwargs);
        double min = arg<double>("min", 3, _args, _kwargs, 0.0);
        double max = arg<double>("max", 4, _args, _kwargs, 0.99 * r0);
        double tol = arg<double>("tol", 5, _args, _kwargs, 0.01 * std::abs(min - max));

        return potential_checkerr(potential_create_well(k, n, r0, tol, min, max));
    }
    catch (const std::exception &e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        return NULL;
    }
    catch(py::error_already_set &e){
        e.restore();
        return NULL;
    }
}


static PyObject *_glj(PyObject *_self, PyObject *_args, PyObject *_kwargs) {
    std::cout << MX_FUNCTION << std::endl;
    
    try {
        double e =   arg<double>("e",   0, _args, _kwargs);
        int m =   arg<double>("m",   1, _args, _kwargs, 3);
        int n =  arg<double>("n",  2, _args, _kwargs, 2*m);
        double k = arg<double>("k", 3, _args, _kwargs, 0);
        double r0 = arg<double>("r0", 4, _args, _kwargs, 1);
        double min = arg<double>("min", 5, _args, _kwargs, 0.05 * r0);
        double max = arg<double>("max", 6, _args, _kwargs, 3 * r0);
        double tol = arg<double>("tol", 7, _args, _kwargs, 0.01);
        bool shifted = arg<bool>("shifted", 8, _args, _kwargs, true);
        
        return potential_checkerr(potential_create_glj(e, n, m, k, r0, min, max, tol, shifted));
    }
    catch (const std::exception &e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        return NULL;
    }
    catch(py::error_already_set &e){
        e.restore();
        return NULL;
    }
}
static PyObject *_overlapping_sphere(PyObject *_self, PyObject *_args, PyObject *_kwargs) {
    std::cout << MX_FUNCTION << std::endl;
    
    try {
        double mu =   arg<double>("mu",   0, _args, _kwargs, 1);
        double kc = arg<double>("kc", 1, _args, _kwargs, 1);
        double kh = arg<double>("kh", 2, _args, _kwargs, 0.0);
        double r0 = arg<double>("r0", 3, _args, _kwargs, 0.0);
        double min = arg<double>("min", 4, _args, _kwargs, 0.001);
        double max = arg<double>("max", 5, _args, _kwargs, 10);
        double tol = arg<double>("tol", 6, _args, _kwargs, 0.001);
        
        return potential_checkerr(potential_create_overlapping_sphere(mu, kc, kh, r0, min, max, tol));
    }
    catch (const std::exception &e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        return NULL;
    }
    catch(py::error_already_set &e){
        e.restore();
        return NULL;
    }
}



static PyObject *_potential_set_value(PyObject *_self, PyObject *_args, PyObject *_kwargs) {
   
    PyObject *key = PyTuple_GetItem(_args, 0);
    PyObject *value = PyTuple_GetItem(_args, 1);
    
    if(key == NULL || value == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "potential_set_dict_value arguments null");
        return NULL;
    }
    
    int result = PyDict_SetItem(MxPotential_Type.tp_dict, key, value);
    
    if(result) {
        return NULL;
    }
    else {
        Py_RETURN_NONE;
    }
}



static PyMethodDef potential_methods[] = {
    {
        "lennard_jones_12_6",
        (PyCFunction)_lennard_jones_12_6,
        METH_VARARGS | METH_KEYWORDS | METH_STATIC,
        "Creates a #potential representing a 12-6 Lennard-Jones potential          \n"
        "                                                                          \n"
        "@param min The smallest radius for which the potential will be constructed. \n"
        "@param max The largest radius for which the potential will be constructed. \n"
        "@param A The first parameter of the Lennard-Jones potential. \n"
        "@param B The second parameter of the Lennard-Jones potential. \n"
        "@param tol The tolerance to which the interpolation should match the exact \n"
        "potential. \n"
        " \n"
        "@return A newly-allocated #potential representing the potential \n"
        "@f$ \left( \frac{A}{r^{12}} - \frac{B}{r^6} \right) @f$ in @f$[a,b]@f$ \n"
        "or @c NULL on error (see #potential_err). \n"
    },
    {
        "lennard_jones_12_6_coulomb",
        (PyCFunction)_lennard_jones_12_6_coulomb,
        METH_VARARGS | METH_KEYWORDS | METH_STATIC,
        ""
    },
    {
        "soft_sphere",
        (PyCFunction)_soft_sphere,
        METH_VARARGS | METH_KEYWORDS | METH_STATIC,
        ""
    },
    {
        "ewald",
        (PyCFunction)_ewald,
        METH_VARARGS | METH_KEYWORDS | METH_STATIC,
        ""
    },
    {
        "coulomb",
        (PyCFunction)_coulomb,
        METH_VARARGS | METH_KEYWORDS | METH_STATIC,
        ""
    },
    {
        "harmonic",
        (PyCFunction)_harmonic,
        METH_VARARGS | METH_KEYWORDS | METH_STATIC,
        ""
    },
    {
        "linear",
        (PyCFunction)_linear,
        METH_VARARGS | METH_KEYWORDS | METH_STATIC,
        ""
    },
    {
        "harmonic_angle",
        (PyCFunction)_harmonic_angle,
        METH_VARARGS | METH_KEYWORDS | METH_STATIC,
        ""
    },
    {
        "harmonic_dihedral",
        (PyCFunction)_harmonic_dihedral,
        METH_VARARGS | METH_KEYWORDS | METH_STATIC,
        ""
    },
    {
        "well",
        (PyCFunction)_well,
        METH_VARARGS | METH_KEYWORDS | METH_STATIC,
        "Square well potential"
    },
    {
        "glj",
        (PyCFunction)_glj,
        METH_VARARGS | METH_KEYWORDS | METH_STATIC,
        "Generalized Lennard-Joned potential"
    },
    {
        "os",
        (PyCFunction)_overlapping_sphere,
        METH_VARARGS | METH_KEYWORDS | METH_STATIC,
        "Overlapping Sphere (soft) potential"
    },
    {
        "overlapping_sphere",
        (PyCFunction)_overlapping_sphere,
        METH_VARARGS | METH_KEYWORDS | METH_STATIC,
       "Overlapping Sphere (soft) potential"
    },
    {
        "_set_dict_value",
        (PyCFunction)_potential_set_value,
        METH_VARARGS | METH_KEYWORDS | METH_STATIC,
        "set dictionary value"
    },
    {
        "force",
        (PyCFunction)potential_force,
        METH_VARARGS | METH_KEYWORDS,
        "calc force"
    },
    
    {NULL}
};


static PyGetSetDef potential_getset[] = {
    {
        .name = "name",
        .get = [](PyObject *_obj, void *p) -> PyObject* {
            MxPotential *obj = (MxPotential*)_obj;
            return carbon::cast(std::string(obj->name));
        },
        .set = [](PyObject *_obj, PyObject *val, void *p) -> int {
            PyErr_SetString(PyExc_PermissionError, "read only");
            return -1;
        },
        .doc = "test doc",
        .closure = NULL
    },
    {
        .name = "min",
        .get = [](PyObject *_obj, void *p) -> PyObject* {
            MxPotential *obj = (MxPotential*)_obj;
            return pybind11::cast(obj->a).release().ptr();
        },
        .set = [](PyObject *_obj, PyObject *val, void *p) -> int {
            PyErr_SetString(PyExc_PermissionError, "read only");
            return -1;
        },
        .doc = "test doc",
        .closure = NULL
    },
    {
        .name = "max",
        .get = [](PyObject *_obj, void *p) -> PyObject* {
            MxPotential *obj = (MxPotential*)_obj;
            return pybind11::cast(obj->b).release().ptr();
        },
        .set = [](PyObject *_obj, PyObject *val, void *p) -> int {
            PyErr_SetString(PyExc_PermissionError, "read only");
            return -1;
        },
        .doc = "test doc",
        .closure = NULL
    },
    {
        .name = "domain",
        .get = [](PyObject *_obj, void *p) -> PyObject* {
            MxPotential *obj = (MxPotential*)_obj;
            py::tuple  res = py::make_tuple(obj->a, obj->b);
            return res.release().ptr();
        },
        .set = [](PyObject *_obj, PyObject *val, void *p) -> int {
            PyErr_SetString(PyExc_PermissionError, "read only");
            return -1;
        },
        .doc = "test doc",
        .closure = NULL
    },
    {
        .name = "intervals",
        .get = [](PyObject *_obj, void *p) -> PyObject* {
            MxPotential *obj = (MxPotential*)_obj;
            return pybind11::cast(obj->n).release().ptr();
        },
        .set = [](PyObject *_obj, PyObject *val, void *p) -> int {
            PyErr_SetString(PyExc_PermissionError, "read only");
            return -1;
        },
        .doc = "test doc",
        .closure = NULL
    },
    {
        .name = "flags",
        .get = [](PyObject *_obj, void *p) -> PyObject* {
            MxPotential *obj = (MxPotential*)_obj;
            return pybind11::cast(obj->flags).release().ptr();
        },
        .set = [](PyObject *_obj, PyObject *val, void *p) -> int {
            MxPotential *obj = (MxPotential*)_obj;
            obj->flags = pybind11::cast<uint32_t>(val);
            return 0;
        },
        .doc = "test doc",
        .closure = NULL
    },
    {
        .name = "bound",
        .get = [](PyObject *_obj, void *p) -> PyObject* {
            MxPotential *obj = (MxPotential*)_obj;
            if(obj->flags & POTENTIAL_BOUND) {
                Py_RETURN_TRUE;
            }
            else {
                Py_RETURN_FALSE;
            }
        },
        .set = [](PyObject *_obj, PyObject *val, void *p) -> int {
            MxPotential *obj = (MxPotential*)_obj;
            if(PyBool_Check(val)) {
                if(val == Py_True) {
                    obj->flags |= POTENTIAL_BOUND;
                }
                else {
                    obj->flags &= ~POTENTIAL_BOUND;
                }
            }
            else {
                PyErr_SetString(PyExc_ValueError, "Potential.bound is a boolean");
            }
            return 0;
        },
        .doc = "test doc",
        .closure = NULL
    },
    {
        .name = "r0",
        .get = [](PyObject *_obj, void *p) -> PyObject* {
            MxPotential *obj = (MxPotential*)_obj;
            return PyFloat_FromDouble(obj->r0);
        },
        .set = [](PyObject *_obj, PyObject *val, void *p) -> int {
            MxPotential *obj = (MxPotential*)_obj;
            if(PyNumber_Check(val)) {
                obj->r0 = PyFloat_AsDouble(val);
            }
            else {
                PyErr_SetString(PyExc_ValueError, "r0 is a number");
                return -1;
            }
            return 0;
        },
        .doc = "test doc",
        .closure = NULL
    },
    {
        .name = "shifted",
        .get = [](PyObject *_obj, void *p) -> PyObject* {
            MxPotential *obj = (MxPotential*)_obj;
            if(obj->flags & POTENTIAL_SHIFTED) {
                Py_RETURN_TRUE;
            }
            else {
                Py_RETURN_FALSE;
            }
        },
        .set = [](PyObject *_obj, PyObject *val, void *p) -> int {
            MxPotential *obj = (MxPotential*)_obj;
            if(PyBool_Check(val)) {
                if(val == Py_True) {
                    obj->flags |= POTENTIAL_SHIFTED;
                }
                else {
                    obj->flags &= ~POTENTIAL_SHIFTED;
                }
            }
            else {
                PyErr_SetString(PyExc_ValueError, "shifted is a boolean");
            }
            return 0;
        },
        .doc = "test doc",
        .closure = NULL
    },
    {
        .name = "r_square",
        .get = [](PyObject *_obj, void *p) -> PyObject* {
            MxPotential *obj = (MxPotential*)_obj;
            if(obj->flags & POTENTIAL_R2) {
                Py_RETURN_TRUE;
            }
            else {
                Py_RETURN_FALSE;
            }
        },
        .set = [](PyObject *_obj, PyObject *val, void *p) -> int {
            MxPotential *obj = (MxPotential*)_obj;
            if(PyBool_Check(val)) {
                if(val == Py_True) {
                    obj->flags |= POTENTIAL_R2;
                }
                else {
                    obj->flags &= POTENTIAL_R2;
                }
            }
            else {
                PyErr_SetString(PyExc_ValueError, "r_square is a boolean");
            }
            return 0;
        },
        .doc = "test doc",
        .closure = NULL
    },
    {NULL}
};


PyTypeObject MxPotential_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "Potential",
    .tp_basicsize = sizeof(MxPotential),
    .tp_itemsize =       0, 
    .tp_dealloc =        potential_dealloc,
    .tp_print =          0, 
    .tp_getattr =        0, 
    .tp_setattr =        0, 
    .tp_as_async =       0, 
    .tp_repr =           0, 
    .tp_as_number =      0, 
    .tp_as_sequence =    0, 
    .tp_as_mapping =     0, 
    .tp_hash =           0, 
    .tp_call =           potential_call,
    .tp_str =            0, 
    .tp_getattro =       0, 
    .tp_setattro =       0, 
    .tp_as_buffer =      0, 
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "Custom objects",
    .tp_traverse =       0, 
    .tp_clear =          0, 
    .tp_richcompare =    0, 
    .tp_weaklistoffset = 0, 
    .tp_iter =           0, 
    .tp_iternext =       0, 
    .tp_methods =        potential_methods,
    .tp_members =        0, 
    .tp_getset =         potential_getset,
    .tp_base =           0, 
    .tp_dict =           0, 
    .tp_descr_get =      0, 
    .tp_descr_set =      0, 
    .tp_dictoffset =     0, 
    .tp_init =           0, 
    .tp_alloc =          [] (struct _typeobject *type, Py_ssize_t n_items) -> PyObject* {

                             if(PyType_IsSubtype(type, &MxPotential_Type) == 0) {
                                 PyErr_SetString(PyExc_ValueError, "MxPotential.tp_alloc can only be used for MxPotential derived objects");
                                 return NULL;
                             }

                             if(type->tp_itemsize != 0 || n_items != 0) {
                                 PyErr_SetString(PyExc_ValueError, "MxPotential.tp_alloc can only be used for single instance potentials");
                                 return NULL;
                             }

                             return (PyObject*)potential_alloc(type);
                         },
    .tp_new =            0, 
    .tp_free =           0, 
    .tp_is_gc =          0, 
    .tp_bases =          0, 
    .tp_mro =            0, 
    .tp_cache =          0, 
    .tp_subclasses =     0, 
    .tp_weaklist =       0, 
    .tp_del =            0, 
    .tp_version_tag =    0, 
    .tp_finalize =       0, 
};


HRESULT _MxPotential_init(PyObject *m)
{
    if (PyType_Ready((PyTypeObject*)&MxPotential_Type) < 0) {
        return E_FAIL;
    }

    Py_INCREF(&MxPotential_Type);
    if (PyModule_AddObject(m, "Potential", (PyObject *)&MxPotential_Type) < 0) {
        Py_DECREF(&MxPotential_Type);
        return E_FAIL;
    }
    

    py::enum_<PotentialFlags>(m, "PotentialFlags", py::arithmetic())
        .value("POTENTIAL_NONE", PotentialFlags::POTENTIAL_NONE)
        .value("POTENTIAL_LJ126", PotentialFlags::POTENTIAL_LJ126)
        .value("POTENTIAL_EWALD", PotentialFlags::POTENTIAL_EWALD)
        .value("POTENTIAL_COULOMB", PotentialFlags::POTENTIAL_COULOMB)
        .value("POTENTIAL_SINGLE", PotentialFlags::POTENTIAL_SINGLE)
        .value("POTENTIAL_R2", PotentialFlags::POTENTIAL_R2)
        .value("POTENTIAL_R", PotentialFlags::POTENTIAL_R)
        .value("POTENTIAL_ANGLE", PotentialFlags::POTENTIAL_ANGLE)
        .value("POTENTIAL_HARMONIC", PotentialFlags::POTENTIAL_HARMONIC)
        .value("POTENTIAL_DIHEDRAL", PotentialFlags::POTENTIAL_DIHEDRAL)
        .value("POTENTIAL_SWITCH", PotentialFlags::POTENTIAL_SWITCH)
        .value("POTENTIAL_REACTIVE", PotentialFlags::POTENTIAL_REACTIVE)
        .value("POTENTIAL_SCALED", PotentialFlags::POTENTIAL_SCALED)
        .value("POTENTIAL_SHIFTED", PotentialFlags::POTENTIAL_SHIFTED)
        .export_values();

    return S_OK;
}


static double potential_create_well_k;
static double potential_create_well_r0;
static double potential_create_well_n;


/* the potential functions */
static double potential_create_well_f ( double r ) {
    return potential_create_well_k/Power(-r + potential_create_well_r0,potential_create_well_n);
}

static double potential_create_well_dfdr ( double r ) {
    return potential_create_well_k * potential_create_well_n *
            Power(-r + potential_create_well_r0,-1 - potential_create_well_n);
}

static double potential_create_well_d6fdr6 ( double r ) {
    return -(potential_create_well_k*(-5 - potential_create_well_n)*
            (-4 - potential_create_well_n)*(-3 - potential_create_well_n)*
            (-2 - potential_create_well_n)*(-1 - potential_create_well_n)*
            potential_create_well_n*
            Power(-r + potential_create_well_r0,-6 - potential_create_well_n));
}


MxPotential *potential_create_well(double k, double n, double r0, double tol, double min, double max)
{
    MxPotential *p = NULL;

    /* allocate the potential */
    if ((p = potential_alloc(&MxPotential_Type)) == NULL ) {
        error(potential_err_malloc);
        return NULL;
    }

    p->flags =  POTENTIAL_R2  | POTENTIAL_LJ126 ;
    p->name = "Well";

    /* fill this potential */
    potential_create_well_k = k;
    potential_create_well_r0 = r0;
    potential_create_well_n = n;

    if (potential_init( p ,
            &potential_create_well_f ,
            &potential_create_well_dfdr ,
            &potential_create_well_d6fdr6 ,
            min , max , tol ) < 0 ) {
        CAligned_Free(p);
        return NULL;
    }

    /* return it */
    return p;
}


static double potential_create_glj_e;
static double potential_create_glj_m;
static double potential_create_glj_n;
static double potential_create_glj_r0;
static double potential_create_glj_k;

/* the potential functions */
static double potential_create_glj_f ( double r ) {
    double e = potential_create_glj_e;
    double n = potential_create_glj_n;
    double m = potential_create_glj_m;
    double r0 = potential_create_glj_r0;
    double k = potential_create_glj_k;
    return k*Power(-r + r0,2) + (e*(-(n*Power(r0/r,m)) + m*Power(r0/r,n)))/(-m + n);
}

static double potential_create_glj_dfdr ( double r ) {
    double e = potential_create_glj_e;
    double n = potential_create_glj_n;
    double m = potential_create_glj_m;
    double r0 = potential_create_glj_r0;
    double k = potential_create_glj_k;
    return 2*k*(-r + r0) + (e*((m*n*r0*Power(r0/r,-1 + m))/Power(r,2) - (m*n*r0*Power(r0/r,-1 + n))/Power(r,2)))/(-m + n);
}

static double potential_create_glj_d6fdr6 ( double r ) {
    double e = potential_create_glj_e;
    double n = potential_create_glj_n;
    double m = potential_create_glj_m;
    double r0 = potential_create_glj_r0;

    return (e*(-(n*(((-5 + m)*(-4 + m)*(-3 + m)*(-2 + m)*(-1 + m)*m*Power(r0,6)*Power(r0/r,-6 + m))/Power(r,12) +
                    (30*(-4 + m)*(-3 + m)*(-2 + m)*(-1 + m)*m*Power(r0,5)*Power(r0/r,-5 + m))/Power(r,11) +
                    (300*(-3 + m)*(-2 + m)*(-1 + m)*m*Power(r0,4)*Power(r0/r,-4 + m))/Power(r,10) +
                    (1200*(-2 + m)*(-1 + m)*m*Power(r0,3)*Power(r0/r,-3 + m))/Power(r,9) +
                    (1800*(-1 + m)*m*Power(r0,2)*Power(r0/r,-2 + m))/Power(r,8) + (720*m*r0*Power(r0/r,-1 + m))/Power(r,7))
                 ) + m*(((-5 + n)*(-4 + n)*(-3 + n)*(-2 + n)*(-1 + n)*n*Power(r0,6)*Power(r0/r,-6 + n))/Power(r,12) +
                        (30*(-4 + n)*(-3 + n)*(-2 + n)*(-1 + n)*n*Power(r0,5)*Power(r0/r,-5 + n))/Power(r,11) +
                        (300*(-3 + n)*(-2 + n)*(-1 + n)*n*Power(r0,4)*Power(r0/r,-4 + n))/Power(r,10) +
                        (1200*(-2 + n)*(-1 + n)*n*Power(r0,3)*Power(r0/r,-3 + n))/Power(r,9) +
                        (1800*(-1 + n)*n*Power(r0,2)*Power(r0/r,-2 + n))/Power(r,8) + (720*n*r0*Power(r0/r,-1 + n))/Power(r,7))))
    /(-m + n);
}


MxPotential *potential_create_glj(double e, double m, double n, double k,
                                  double r0, double min, double max,
                                  double tol, bool shifted)
{
    MxPotential *p = NULL;
    
    /* allocate the potential */
    if ((p = potential_alloc(&MxPotential_Type)) == NULL ) {
        error(potential_err_malloc);
        return NULL;
    }
    
    p->flags =  POTENTIAL_R2  | POTENTIAL_LJ126 | POTENTIAL_SCALED;
    p->name = "Generalized Lennard-Jones";
    
    /* fill this potential */
    potential_create_glj_e = e;
    potential_create_glj_n = n;
    potential_create_glj_m = m;
    potential_create_glj_r0 = r0;
    potential_create_glj_k = k;
    
    if (potential_init(p ,
                       &potential_create_glj_f ,
                       &potential_create_glj_dfdr ,
                       &potential_create_glj_d6fdr6 ,
                       min , max , tol ) < 0 ) {
        CAligned_Free(p);
        return NULL;
    }
    
    if(shifted) {
        p->r0 = r0;
        p->flags &= ~POTENTIAL_SCALED;
        p->flags |= POTENTIAL_SHIFTED;
    }
    
    /* return it */
    return p;
}

int MxPotential_Check(PyObject *obj) {
    return PyObject_IsInstance(obj, (PyObject*)&MxPotential_Type);
}
