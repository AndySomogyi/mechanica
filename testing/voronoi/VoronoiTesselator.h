/*
 * VoronoiTesselator.h
 *
 *  Created on: Jul 7, 2017
 *      Author: andy
 */

#ifndef TESTING_VORONOI_VORONOITESSELATOR_H_
#define TESTING_VORONOI_VORONOITESSELATOR_H_

#include <Magnum/Buffer.h>
#include <Magnum/DefaultFramebuffer.h>
#include <Magnum/Mesh.h>
#include <Magnum/Math/Vector3.h>
#include <Magnum/Platform/GlfwApplication.h>
#include <Magnum/Shaders/VertexColor.h>
#include <Magnum/Primitives/Cube.h>
#include <MagnumPlugins/AssimpImporter/AssimpImporter.h>
#include <Magnum/Trade/MeshData3D.h>
#include <vector>

#include <MxMesh.h>


using namespace Magnum;
using namespace Magnum::Trade;


class VoronoiTesselator {
public:
    static std::optional<MeshData3D> tesselate(const std::vector<Vector3>& positions,
            const Vector3& min, const Vector3& max, const Vector3ui& partitions);
};

#endif /* TESTING_VORONOI_VORONOITESSELATOR_H_ */
