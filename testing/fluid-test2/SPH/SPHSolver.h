/*
    This file is part of Mechanica.

    Based on Magnum example

    Original authors — credit is appreciated but not required:

        2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019 —
            Vladimír Vondruš <mosra@centrum.cz>
        2019 — Nghia Truong <nghiatruong.vn@gmail.com>

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.
 */


#pragma once

#include <vector>
#include <Magnum/Magnum.h>

#include "SPH/SPHKernels.h"
#include "SPH/DomainBox.h"


struct SPHParams {
    Float stiffness = 20000.0f;
    Float viscosity = 0.05f;
    Float boundaryRestitution = 0.5f;
};

/* This is a very basic implementation of SPH (Smoothed Particle Hydrodynamics)
   solver. For the purpose of fast running (as this is a real-time application
   example), accuracy has been heavily sacrificed for performance. */
class SPHSolver {
    public:
        explicit SPHSolver(Float particleRadius);

        void setPositions(const std::vector<Vector3>& particlePositions);
        void reset();
        void advance();

        DomainBox& domainBox() { return _domainBox; }
        SPHParams& simulationParameters() { return _params; }

        std::size_t numParticles() const { return _positions.size(); }
        const std::vector<Vector3>& particlePositions() { return _positions; }

    private:
        void computeDensities();
        void velocityIntegration(Float timestep);
        void computeViscosity();
        void updatePositions(Float timestep);

        /* Rest density of fluid */
        constexpr static Float RestDensity = 1000.0f;

        /* Particle radius = half distance between consecutive particles */
        const Float _particleRadius;
        /* particle_mass = pow(particle_spacing, 3) * rest_density */
        const Float _particleMass;

        std::vector<Vector3> _positions;
        std::vector<Vector3> _positionsT0; /* Initial positions */

        /* For saving memory thus gaining performance,
           particle densities (float32) are clamped to [0, 10000] then cast
           into uint16_t */
        std::vector<uint16_t> _densities;

        /* Other particle states */
        std::vector<std::vector<uint32_t>> _neighbors;
        std::vector<std::vector<Vector3>>  _relPositions;
        std::vector<Vector3> _velocities;
        std::vector<Vector3> _velocityDiffusions;

        /* SPH kernels */
        SPHKernels _kernels;

        /* Boundary */
        DomainBox _domainBox;

        /* Parameters */
        SPHParams _params;
};


