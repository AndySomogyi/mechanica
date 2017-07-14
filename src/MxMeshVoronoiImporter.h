/*
 * MxMeshVoronoiImporter.h
 *
 *  Created on: Jul 8, 2017
 *      Author: andy
 */

#ifndef SRC_MXMESHVORONOIIMPORTER_H_
#define SRC_MXMESHVORONOIIMPORTER_H_

#include "MxMesh.h"
#include <Magnum/Math/Vector3.h>
#include <vector>
#include <array>

class MxMeshVoronoiImporter {
public:

    /** The class constructor sets up the geometry of container.
     * \param[in] (ax_,bx_) the minimum and maximum x coordinates.
     * \param[in] (ay_,by_) the minimum and maximum y coordinates.
     * \param[in] (az_,bz_) the minimum and maximum z coordinates.
     * \param[in] (nx_,ny_,nz_) the number of grid blocks in each of the three
     *                       coordinate directions.
     * \param[in] (xperiodic_,yperiodic_,zperiodic_) flags setting whether the
     *                                               container is periodic in each
     *                                               coordinate direction.
     * \param[in] init_mem the initial memory allocation for each block. */
    //container::container(double ax_,double bx_,double ay_,double by_,double az_,double bz_,
    //    int nx_,int ny_,int nz_,bool xperiodic_,bool yperiodic_,bool zperiodic_,int init_mem)
    static bool readFile(const std::string& path, const Magnum::Vector3 &min,
            const Magnum::Vector3 &max, const Magnum::Vector3i &n,
            const std::array<bool, 3> periodic, MxMesh& mesh);

    static bool testing(const std::string& path, const Magnum::Vector3 &min,
                const Magnum::Vector3 &max, const Magnum::Vector3i &n,
                const std::array<bool, 3> periodic, MxMesh& mesh);

    static bool random(uint numPts, const Magnum::Vector3 &min,
                const Magnum::Vector3 &max, const Magnum::Vector3i &n,
                const std::array<bool, 3> periodic, MxMesh& mesh);


};

#endif /* SRC_MXMESHVORONOIIMPORTER_H_ */
