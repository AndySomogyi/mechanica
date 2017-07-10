/*
 * VoronoiTesselator.cpp
 *
 *  Created on: Jul 7, 2017
 *      Author: andy
 */

#include "VoronoiTesselator.h"
#include "voro++.hh"

std::optional<MeshData3D> VoronoiTesselator::tesselate(
        const std::vector<Vector3>& positions,
        const Vector3& min, const Vector3& max, const Vector3ui& partitions) {

    // container(double ax_,double bx_,double ay_,double by_,double az_,double bz_,
    // int nx_,int ny_,int nz_,bool xperiodic_,bool yperiodic_,bool zperiodic_,int init_mem);

    voro::container container(min.x(), max.x(), min.y(), max.y(), min.z(), max.z(),
            partitions.x(), partitions.y(), partitions.z(), false, false, false, positions.size());

    for (int i = 0; i < positions.size(); ++i) {
        container.put(i, positions[i][0], positions[i][1], positions[i][2]);
    }

    container.compute_all_cells();

    //container.

    return {};


}
