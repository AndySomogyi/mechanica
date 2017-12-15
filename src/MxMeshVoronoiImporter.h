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

     MxMeshVoronoiImporter(MxMesh& mesh, float density = 1.0) :
        mesh{mesh}, density{density}, cellId{mesh.rootCell()->id + 1} {};


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
    bool readFile(const std::string& path, const Magnum::Vector3 &min,
            const Magnum::Vector3 &max, const Magnum::Vector3i &n,
            const std::array<bool, 3> periodic);

    bool testing(const std::string& path, const Magnum::Vector3 &min,
                const Magnum::Vector3 &max, const Magnum::Vector3i &n,
                const std::array<bool, 3> periodic);

    bool random(uint numPts, const Magnum::Vector3 &min,
                const Magnum::Vector3 &max, const Magnum::Vector3i &n,
                const std::array<bool, 3> periodic);


    bool monodisperse();

    bool irregular(MxMesh& mesh);

private:

    MxMesh &mesh;

    const float density;

    uint32_t triId = 0;

    uint32_t cellId = 0;

       /**
     * Search for an existing triangle, if so, attach it to the cell. If not, create
     * a new triangle, and attach it to the cell.
     */
    void createTriangleForCell(const std::array<VertexPtr, 3> &verts, CellPtr cell);


    template<typename T>
    friend bool pack(MxMeshVoronoiImporter& voro, T& container);

};

#endif /* SRC_MXMESHVORONOIIMPORTER_H_ */
