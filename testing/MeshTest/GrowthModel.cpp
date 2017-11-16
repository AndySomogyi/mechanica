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


static struct RedCellType : MxCellType
{
    RedCellType() {
        color = Color4{1.0f, 0.0f, 0.0f, 0.5f};
    }
} redCellType;

static struct BlueCellType : MxCellType
{
    BlueCellType() {
        color = Color4{0.0f, 0.0f, 1.0f, 0.5f};
    };
} blueCellType;

GrowthModel::GrowthModel()  {

    mesh = new MxMesh();

    /*
    MxMeshGmshImporter importer{*mesh,
        [](Gmsh::ElementType, int id) {
            if((id % 2) == 0) {
                return (MxCellType*)&redCellType;
            } else {
                return (MxCellType*)&blueCellType;
            }
        }
    };
    mesh->shortCutoff = 0;
    mesh->longCutoff = 0.1;
    importer.read("/Users/andy/src/mechanica/testing/gmsh1/sheet.msh");
    minTargetArea = 0.001;
    targetArea = 0.3;
    maxTargetArea = 0.5;

    targetVolume = 0.1;
    minTargetVolume = 0.005;
    maxTargetVolume = 0.2;
    */
     
    
    
   

    MxMeshGmshImporter importer{*mesh,
        [](Gmsh::ElementType, int id) {
            if((id % 2) == 0) {
                return (MxCellType*)&redCellType;
            } else {
                return (MxCellType*)&blueCellType;
            }
        }
    };
    mesh->shortCutoff = 0.5;
    mesh->longCutoff = 1.0;
    importer.read("/Users/andy/src/mechanica/testing/MeshTest/simplesheet.msh");
    minTargetArea = 0.001;
    targetArea = 0.1;
    maxTargetArea = 2;

    targetVolume = 0.9;
    minTargetVolume = 0.005;
    maxTargetVolume = 1.5;
     
     





    //importer.read("/Users/andy/src/mechanica/testing/MeshTest/flatcube.msh");

/*
    // cube



    
    MxMeshGmshImporter importer{*mesh, [](Gmsh::ElementType, int id) {return nullptr;}};
    mesh->shortCutoff = 0;
    mesh->longCutoff = 0.6;
    importer.read("/Users/andy/src/mechanica/testing/MeshTest/cube.msh");
    minTargetArea = 0.1;
    targetArea = 6.0;
    maxTargetArea = 15;

    targetVolume = 1.5;
    minTargetVolume = 0.5;
    maxTargetVolume = 25.0;
 */
    
     testEdges();
    
}

//const float targetArea = 0.45;

//const float targetArea = 2.0;




HRESULT GrowthModel::calcForce(TrianglePtr* triangles, uint32_t len) {

    HRESULT result;
    
    testEdges();


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


            //std::cout << "id: " << tri->id << ",AR " << tri->aspectRatio << std::endl;

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
                tri->vertices[v]->force +=  4.0 * diff * (tri->area / cell->area) *  dir[v].normalized();
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

void GrowthModel::testEdges() {
    
    return;
    
    /*
    for(int i = 0; i < mesh->cells.size(); ++i) {
        if(i % 2) {
            mesh->cells[i]->render = true;
        } else {
            mesh->cells[i]->render = false;
        }
    }
     */
    
    for (auto tri : mesh->triangles) {
        tri->alpha = 0.001;
    }
    
    
    for (auto tri : mesh->triangles) {
        
        for(int i = 0; i < 3; ++i) {
            MxEdge e{tri->vertices[i], tri->vertices[(i+1)%3]};
            auto triangles = e.radialTriangles();
            

            for(auto tri : triangles) {
                if(triangles.size() >= 4)
                    tri->alpha = 0.3;
            }
        }
    }
    
}
