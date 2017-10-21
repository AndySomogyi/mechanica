/*
 * GrowthModel.cpp
 *
 *  Created on: Oct 13, 2017
 *      Author: andy
 */

#include "GrowthModel.h"
#include <MxMeshGmshImporter.h>

GrowthModel::GrowthModel()  {

    mesh = new MxMesh();

    MxMeshGmshImporter importer{*mesh};

    //importer.read("/Users/andy/src/mechanica/testing/gmsh1/sheet.msh");

    importer.read("/Users/andy/src/mechanica/testing/growth1/cube.msh");

}

const float targetArea = 2.0;

HRESULT GrowthModel::calcForce(TrianglePtr* triangles, uint32_t len) {

    for(VertexPtr vert : mesh->vertices) {
        vert->force = Vector3{};
    }

    for(uint i = 0; i < len; ++i) {
        TrianglePtr tri = triangles[i];

        float diff = targetArea - tri->area;

        for(int v = 0; v < 3; ++v) {
            tri->vertices[v]->force += diff * (tri->vertices[v]->position - tri->centroid).normalized();
        }
    }


    return S_OK;
}
