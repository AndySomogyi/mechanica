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

    importer.read("/Users/andy/src/mechanica/testing/gmsh1/sheet.msh");

    //importer.read("/Users/andy/src/mechanica/testing/growth1/cube.msh");

}

const float targetArea = 0.75;

HRESULT GrowthModel::calcForce(TrianglePtr* triangles, uint32_t len) {

    HRESULT result;

    for(VertexPtr vert : mesh->vertices) {
        vert->force = Vector3{};
    }



    for(CellPtr cell : mesh->cells) {
        if((result = cellAreaForce(cell)) != S_OK) {
            return mx_error(result, "cell area force");
        }
    }

    return S_OK;
}

HRESULT GrowthModel::cellAreaForce(CellPtr cell) {

    if(mesh->rootCell() == cell) {
        return S_OK;
    }

    float diff = targetArea - cell->area;

    for(auto f: cell->facets) {
        for(auto tri : f->triangles) {

            for(int v = 0; v < 3; ++v) {
                Vector3 dir = tri->vertices[v]->position - tri->centroid;
                float len = dir.length();
                tri->vertices[v]->force += diff * len * dir;
            }
        }
    }
    return S_OK;
}
