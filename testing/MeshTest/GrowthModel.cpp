/*
 * GrowthModel.cpp
 *
 *  Created on: Oct 13, 2017
 *      Author: andy
 */

#include "GrowthModel.h"
#include <MxMeshGmshImporter.h>
#include <MxDebug.h>
#include <iostream>

GrowthModel::GrowthModel()  {

    mesh = new MxMesh();

    MxMeshGmshImporter importer{*mesh};
    
    /*
    
    mesh->shortCutoff = 0;
    mesh->longCutoff = 10;
    importer.read("/Users/andy/src/mechanica/testing/gmsh1/sheet.msh");
    minTargetArea = 0.001;
    targetArea = 0.45;
    maxTargetArea = 0.7;
    
    targetVolume = 0.017;
    minTargetVolume = 0.005;
    maxTargetVolume = 0.03;
     
     */
     
    
    
    

    //importer.read("/Users/andy/src/mechanica/testing/MeshTest/flatcube.msh");
    
    
    // cube
    
    
    mesh->shortCutoff = 0;
    mesh->longCutoff = 0.6;
    importer.read("/Users/andy/src/mechanica/testing/MeshTest/cube.msh");
    minTargetArea = 0.1;
    targetArea = 6.0;
    maxTargetArea = 15;
    
    targetVolume = 1.5;
    minTargetVolume = 0.5;
    maxTargetVolume = 25.0;
     
    
    
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
        
        if((result = cellVolumeForce(cell)) != S_OK) {
            return mx_error(result, "cell volume force");
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
                tri->vertices[v]->force +=  2.0 * diff * (tri->area / cell->area) *  dir[v].normalized();
            }
        }
    }
    return S_OK;
}

HRESULT GrowthModel::cellVolumeForce(CellPtr cell)
{
    //return S_OK;
    
    if(mesh->rootCell() == cell) {
        return S_OK;
    }
    
    assert(cell->area >= 0);
    
    float diff = targetVolume - cell->volume;
    
    for(auto f: cell->facets) {
        for(auto tri : f->triangles) {
            
            Vector3 force = 0.5 * tri->normal * tri->area * diff;
            
            for(int v = 0; v < 3; ++v) {
                tri->vertices[v]->force +=  force;
            }
        }
    }
    

    return S_OK;
}

