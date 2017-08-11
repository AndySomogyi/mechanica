/*
 * MxForceFunction.h
 *
 *  Created on: Jul 22, 2017
 *      Author: andy
 */

#ifndef SRC_MXFORCEFUNCTION_H_
#define SRC_MXFORCEFUNCTION_H_

#include "MxExecutionContext.h"


/**
 * The simulator uses force function to calculate the total force applied to the geometric
 * physical objects (vertices, triangles, etc...). A compiled model (typically read from
 * a compiled shared library) needs to provide 'force functions' to the simulator.
 *
 * The basic simulation flow is:
 *  * Calculate total force applied to each geometric physical object using the specified
 *    MxForceFunction derived objects.
 *  * Use the total force to calculate time evolution (either F \propto ma or F \propto mv).
 *  * update the (possibly velocities) and positions.
 *
 * *Different kinds of force calculations*
 *
 * Molecular dynamics typically define
 *
 * bonds, angles, dihedrals and impropers.
 *
 * The standard bond is essentially a spring that connects two vertices. We know that this
 * spring acts uniformly on each atom in the pair. We can optimize this calculation by performing
 * the calculation only once, but applying it to both vertices at the same time (in opposite
 * directions of course).
 *
 *
 *
 * Volume Energy
 * Acts in the opposite direction of the surface normal of vertex or surface triangle.
 * This is a function of a given surface location and the cell that the surface element
 * belongs to.
 *
 *
 *
 * Surface Energy
 * The net force due to surface tension of a perfectly flat sheet is zero at every interior
 * point. There is no motion of surface elements on sheets at held in a fixed positions,
 * hence the surface tension force sums to zero. On curved surfaces, surface tension acts to
 * pull every surface element towards their neighboring surface elements. On a sphere, force
 * due to surface tension points exactly towards the center of the sphere. Surface tension
 * will tend to make the surface locally flat.
 *
 * Bending Energy
 *
 *
 *
 */


typedef void (*MxForceFunction1Vec3f)(struct ExecutionContext *context, float *result);

typedef void (*MxForceFunction2Vec3f)(struct ExecutionContext *context, float *result1, float *result2);


class MxForceFunction {
};


class MxTriangleForceFunction {

};

#endif /* SRC_MXFORCEFUNCTION_H_ */
