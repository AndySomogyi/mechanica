/*
 * GrowthModel.cpp
 *
 *  Created on: Oct 13, 2017
 *      Author: andy
 */

#include "GrowthModel.h"
#include <MxMeshGmshImporter.h>
#include <iostream>

GrowthModel::GrowthModel()  {

    mesh = new MxMesh();

    MxMeshGmshImporter importer{*mesh};

    //importer.read("/Users/andy/src/mechanica/testing/gmsh1/sheet.msh");

    importer.read("/Users/andy/src/mechanica/testing/MeshTest/cube.msh");

    //importer.read("/Users/andy/src/mechanica/testing/MeshTest/flatcube.msh");
    
    minTargetArea = 0.1;
    targetArea = 2.0;
    maxTargetArea = 10;
    
    

}

//const float targetArea = 0.45;

//const float targetArea = 2.0;




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

    //return S_OK;

    if(mesh->rootCell() == cell) {
        return S_OK;
    }

    assert(cell->area >= 0);

    float diff = targetArea - cell->area;

    for(auto f: cell->facets) {
        for(auto tri : f->triangles) {

            assert(tri->area > 0);
            
            
            
            //if (tri->area > 0.1) {
            //    continue;
            //}
            
            //if(tri->id != 0) {
            //    continue;
            //}
            
            std::cout << "id: " << tri->id << ",AR " << tri->aspectRatio << std::endl;

            Vector3 dir[3];
            float len[3];
            float totLen = 0;
            for(int v = 0; v < 3; ++v) {
                dir[v] = tri->vertices[v]->position - tri->centroid;
                //dir[v] = ((tri->vertices[v]->position - tri->vertices[(v+1)%3]->position) +
                //          (tri->vertices[v]->position - tri->vertices[(v+2)%3]->position)) / 2;
                len[v] = dir[v].length();
                //dir[v] = dir[v].normalized();
                totLen += len[v];
            }

            for(int v = 0; v < 3; ++v) {
                //tri->vertices[v]->force +=  1/3. * diff * (tri->area / cell->area) *  dir[v] / totLen ;
                tri->vertices[v]->force +=  1/3. * diff * (tri->area / cell->area) *  dir[v].normalized();
            }
        }
    }
    return S_OK;
}
