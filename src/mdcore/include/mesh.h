/*******************************************************************************
 * This file is part of mdcore.
 * Coypright (c) 2017 Andy Somogyi (andy.somogyi*at*indiana.edu)
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
 * Created on: Jun 3, 2017
 ******************************************************************************/


#ifndef _MDCORE_INCLUDE_MESH_H_
#define _MDCORE_INCLUDE_MESH_H_

#include "platform.h"


/**
 * triangles are shared so allocate them in a single location, 
 * everything else references them. 
 */
typedef struct triangle {
    /**
     * three particles define the corner vertices of a triangle,
     * these are particle ids.
     */
    int particles[3];
};


/**
 * Each region has a manifold boundary consisting of a set of 
 * partial triangles. 
 *
 * Alternate approaches were to store the ids of the neiboring p_triangle
 * in each triangle. However that approach would lead to confusion when a
 * triangle is shared between regions, i.e. how to find the neighboring regions
 * from a current region. 
 * 
 * Winding direction. Once a triangle is attached to a region bounding mesh, that
 * triangle can not re-orient face-in or face out. It is important to keep track of the
 * winding direction to know what is the front and back face, say we want to render 
 * only the insides or outsides. So, we need to add a field to the p tri struct
 * to specify what order to draw the triangle vertices. This flag is set only
 * when a p_triangle gets attached to a tri. 
 */
typedef struct p_triangle {

    /** index to the mesh's triangle buffer */
    int tri_id;

    /** each p_triangle is adjacent to 3 other triangles */
    int adj_ptri[3];

    /** id of neighboring region */
    int adj_reg_id;

    /** id of sibling partial triangle */
    int sib_ptri_id; 

} p_triangle;

/**
 * 
 */
typedef struct region_3d {
    /** array of surface partial triangle ids */
    int *surface_ptri_ids;
    int surface_ptri_size, surface_ptri_count;
    
} region_3d;



/**
 * The mesh structure
 */
typedef struct mesh {

    /**
     * All triangles are stored int this array. 
     *
     * Note, future versions will be more cache optimal if we store triangles in
     * each space_cell instead of all here. 
     */
    struct traingle *triangles;
    int triangles_size, triangles_count;

    struct p_triangle *p_triangles;
    int p_triangles_size, p_triangles_count;

    struct region *regions;
    int regions_size, regions_count;
    
    
};



MDCORE_BEGIN_DECLS



MDCORE_END_DECLS

#endif /* _MDCORE_INCLUDE_MESH_H_ */
