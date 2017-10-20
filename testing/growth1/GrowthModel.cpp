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
