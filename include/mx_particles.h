/*
 * mx_particles.h
 *
 *  Created on: Feb 1, 2017
 *      Author: andy
 */

#ifndef INCLUDE_MX_PARTICLES_H_
#define INCLUDE_MX_PARTICLES_H_

/**
 * Represents a collection of simple particles.
 *
 * Simple particles are spherical, and are soft-spheres. They have mass (inverse mass),
 * and might have orientation vector.
 *
 * All particles are stored in a MxParticles structure. One can not directly access an
 * indiviaual particle object, but rather all attributes are accessed through the MxParticles
 * collection. Particles are accessed by an index.
 *
 *
 * Scalar fields (such as chemical concentration fields) map a location in space to a scalar
 * value.
 */

CAPI_STRUCT(MxParticles);

#ifdef __cplusplus
extern "C" {
#endif
    
    
CObject *TestParticles_New();

void TestParticles_Step();




    
    

    
    
#ifdef __cplusplus
}
#endif



#endif /* INCLUDE_MX_PARTICLES_H_ */
