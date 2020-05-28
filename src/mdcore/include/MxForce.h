/*
 * MxForce.h
 *
 *  Created on: May 21, 2020
 *      Author: andy
 */

#ifndef SRC_MDCORE_SRC_MXFORCE_H_
#define SRC_MDCORE_SRC_MXFORCE_H_

#include "platform.h"
#include "fptype.h"
#include "carbon.h"
#include <Magnum/Magnum.h>
#include <Magnum/Math/Vector4.h>


enum MXFORCE_KIND {
    MXFORCE_ONEBODY,
    MXFORCE_PAIRWISE
};

/**
 * single body force function.
 */
typedef void (*MxForce_OneBodyPtr)(struct MxForce*, struct MxParticle *, FPTYPE*f);


struct MxForce : PyObject
{
    MxForce_OneBodyPtr func;
};

#endif /* SRC_MDCORE_SRC_MXFORCE_H_ */



/**
 * Old Notes, kept, for now, for reference
 *
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

CAPI_DATA(PyTypeObject) MxForce_Type;

/**
 * internal method, init the Force type, and the forces module and add it to the main module.
 */
HRESULT MXForces_Init(PyObject *m);

