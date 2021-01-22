
/*******************************************************************************
 * This file is part of Mechanica.
 * Coypright (c) 2021 Andy Somogyi (somogyie at indiana dot edu)
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

#pragma once
#ifndef _MDCORE_CUBOID_EVAL_H_
#define _MDCORE_CUBOID_EVAL_H_

#include "MxPotential.h"
#include "MxCuboid.hpp"
#include "potential_eval.h"
#include "engine.h"
#include "space_cell.h"

/**
 * Cuboids will typically be large, proportional to the size of a
 * space cell, first check if AABB of cuboid and space cell overlap.
 *
 * The cuboid AABB *includes* the cutoff distance.
 */
INLINE bool aabb_intersect_cuboid_spacecell(MxCuboid *cuboid, struct space_cell *c) {
    Magnum::Vector3 a_min = cuboid->aabb.min();
    Magnum::Vector3 a_max = cuboid->aabb.max();
    Magnum::Vector3 b_min = {
        float(c->origin[0]) - (float)_Engine.s.cutoff,
        float(c->origin[1]) - (float)_Engine.s.cutoff,
        float(c->origin[2]) - (float)_Engine.s.cutoff
    };
    Magnum::Vector3 b_max = {
        float(c->origin[0]) + (float)c->dim[0] + (float)_Engine.s.cutoff,
        float(c->origin[1]) + (float)c->dim[1] + (float)_Engine.s.cutoff,
        float(c->origin[2]) + (float)c->dim[2] + (float)_Engine.s.cutoff
    };

    return (a_min[0] <= b_max[0] && a_max[0] >= b_min[0]) &&
           (a_min[1] <= b_max[1] && a_max[1] >= b_min[1]) &&
           (a_min[2] <= b_max[2] && a_max[2] >= b_min[2]);
}


/**
 * checks aabb collision between a cuboid with global coords, and particle in a cell with
 * cell local coords
 */
INLINE bool aabb_intersect_cuboid_particle(MxCuboid *cuboid, MxParticle *p, struct space_cell *c) {
    Magnum::Vector3 box_min = cuboid->aabb.min() - Magnum::Vector3(_Engine.s.cutoff);
    Magnum::Vector3 box_max = cuboid->aabb.max() + Magnum::Vector3(_Engine.s.cutoff);
    Magnum::Vector3 point = {
        p->position[0] + float(c->origin[0]),
        p->position[1] + float(c->origin[1]),
        p->position[2] + float(c->origin[2])
    };
    
    return (point[0] >= box_min[0] && point[0] <= box_max[0]) &&
           (point[1] >= box_min[1] && point[1] <= box_max[1]) &&
           (point[2] >= box_min[2] && point[2] <= box_max[2]);
}

INLINE bool potential_eval_cuboid_particle(MxCuboid *cube, MxParticle *part, struct space_cell *cell) {
    
    MxPotential *pot = _Engine.cuboid_potentials[part->typeId];
    
    if(!pot) {
        return false;
    }
    
    Magnum::Vector3 box_min = cube->aabb.min() - Magnum::Vector3(_Engine.s.cutoff);
    Magnum::Vector3 box_max = cube->aabb.max() + Magnum::Vector3(_Engine.s.cutoff);
    
    // point in global space
    Magnum::Vector3 point = {
        part->position[0] + float(cell->origin[0]),
        part->position[1] + float(cell->origin[1]),
        part->position[2] + float(cell->origin[2])
    };
    
    // check AABB
    if(!((point[0] >= box_min[0] && point[0] <= box_max[0]) &&
         (point[1] >= box_min[1] && point[1] <= box_max[1]) &&
         (point[2] >= box_min[2] && point[2] <= box_max[2]))) {
        return false;
    }
    
    // transform point into cuboid local coordinate space
    
    // translate into box local coordinate space:
    // global = part.pos + cell.origin
    // box.local = global - box.origin
    // -> box.local = part.pos + cell.origin - box.origin
    point = point - cube->position;
    
    point = cube->inv_orientation.transformVector(point);
    
    // absolute value of point, only work in octant 1.
    Magnum::Vector3 absPoint = {std::abs(point[0]), std::abs(point[1]), std::abs(point[2])};
    
    Magnum::Vector3 halfSize = cube->size / 2;
    
    // distance between point and box surface, negative if inside the box.
    Magnum::Vector3 distanceVec = absPoint - halfSize;
    Magnum::Vector3 forceVec = {0, 0, 0};
    
    float r0, distance, force, energy;
    
    // inside the box
    if(distanceVec[0] < 0 && distanceVec[1] < 0 && distanceVec[2] < 0 ) {
        if(distanceVec[0] > distanceVec[1] && distanceVec[0] > distanceVec[2] ) {
            r0 = halfSize[0];
            forceVec = {point[0], 0, 0};
            distance = std::abs(point[0]);
        }
        else if(distanceVec[1] > distanceVec[0] && distanceVec[1] > distanceVec[2] ) {
            r0 = halfSize[1];
            forceVec = {0, point[1], 0};
            distance = std::abs(point[1]);
        }
        else {
            r0 = halfSize[2];
            forceVec = {0, 0, point[2]};
            distance = std::abs(point[2]);
        }
    }
    
    // above the box in x-y plane (force in z direction)
    else if(distanceVec[0] < 0 && distanceVec[1] < 0) {
        r0 = halfSize[2];
        forceVec = {0, 0, point[2]};
        distance = std::abs(point[2]);
    }
    
    // check if in x-z plane (force in y direction)
    else if(distanceVec[0] < 0 && distanceVec[2] < 0) {
        r0 = halfSize[1];
        forceVec = {0, point[1], 0};
        distance = std::abs(point[1]);
    }
    
    // check if in y-z plane (force in x direction)
    else if(distanceVec[1] < 0 && distanceVec[2] < 0) {
        r0 = halfSize[0];
        forceVec = {point[0], 0, 0};
        distance = std::abs(point[0]);
    }
    
    else {
        return false;
    }

    /* update the forces if part in range */
    
    // resulting force from potential_eval is force scalar divided by total distance
    // for central potentials, this lets create a normalized unit vector direction.
    if (potential_eval_ex(pot, r0, part->radius, distance * distance, &energy, &force)) {
        
        //        for ( int k = 0 ; k < 3 ; k++ ) {
        //            w = force * dx[k];
        //            pif[k] -= w;
        //            // TODO large parts frozen for now
        //            //part_j->f[k] += w;
        //        }
        //        /* tabulate the energy */
        //        epot += e;
        forceVec = -forceVec * force;
        
        // transform force vector in local space to global space
        forceVec = cube->orientation.transformVector(forceVec);
        
        part->force += forceVec;
        
        return true;
    }
    
    
    return false;
}

inline std::vector<MxCuboid*> space_cell_intersecting_cuboids(struct space_cell *c) {
    std::vector<MxCuboid*> cuboids;
    for (MxCuboid& cuboid : _Engine.s.cuboids) {
        if(aabb_intersect_cuboid_spacecell(&cuboid, c)) {
            aabb_intersect_cuboid_spacecell(&cuboid, c);
            cuboids.push_back(&cuboid);
        }
    }
    return cuboids;
}


#endif
