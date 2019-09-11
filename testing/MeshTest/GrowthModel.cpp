/*
 * GrowthModel.cpp
 *
 *  Created on: Oct 13, 2017
 *      Author: andy
 */

#include "GrowthModel.h"
#include <MxDebug.h>
#include <iostream>
#include "MeshIO.h"
#include "MeshTest.h"
#include "T1Transition.h"
#include "T2Transition.h"
#include "T3Transition.h"
#include "MxCellVolumeConstraint.h"
#include "MxPolygonSurfaceTensionForce.h"


static struct RedCellType : MxCellType
{
    virtual Magnum::Color4 color(struct MxCell *cell) {
        //return Color4{1.0f, 0.0f, 0.0f, 0.08f};
        return Color4::green();
    }

    virtual ~RedCellType() {};

    RedCellType() : MxCellType{"RedCellType", MxCell_Type} {};

} redCellType;

static struct BlueCellType : MxCellType
{
    virtual Magnum::Color4 color(struct MxCell *cell) {
        return Color4{0.0f, 0.0f, 1.0f, 0.08f};
    }

    virtual ~BlueCellType() {};

    BlueCellType() : MxCellType{"BlueCellType", MxCell_Type} {};

} blueCellType;

static struct ClearCellType : MxCellType
{
    virtual Magnum::Color4 color(struct MxCell *cell) {
        switch(cell->id) {
            case 4: return Color4{0.0f, 0.0f, 1.0f, 0.08f};
            default: return Color4{0.0f, 0.0f, 0.0f, 0.00f};
        }
    }

    virtual ~ClearCellType() {};

    ClearCellType() : MxCellType{"ClearCellType", MxCell_Type} {};

} clearCellType;

static struct MeshObjectTypeHandler : IMeshObjectTypeHandler {
    virtual MxType *cellType(const char* cellName, int cellIndex) {
        return &redCellType;
    }

    virtual MxType *polygonType(int polygonIndex) {
        return nullptr;
    }

    virtual MxType *partialPolygonType(const MxType *cellType, const MxType *polyType) {
        return nullptr;
    }

    virtual ~MeshObjectTypeHandler() {};

} meshObjectTypeHandler;


MxCellVolumeConstraint cellVolumeConstraint{0., 0.};

MxPolygonSurfaceTensionForce stdPolygonForce{0.05};


GrowthModel::GrowthModel()  {
}


HRESULT GrowthModel::loadModel() {
    loadAssImpModel();

    for(int i = 0; i < mesh->cells.size(); ++i) {
        CellPtr cell = mesh->cells[i];
        std::cout << "cell[" << i << "], id:" << cell->id << ", center: " << cell->centroid << std::endl;
    }

    testEdges();

    VERIFY(propagator->structureChanged());

    return S_OK;
}


void GrowthModel::loadAssImpModel() {

    const std::string dirName = "/Users/andy/src/mechanica/testing/models/";

    //const char* fileName = "football.t1.obj";
    //const char* fileName = "football.t2.obj";
    const char* fileName = "t2.test1.obj";
    //const char* fileName = "cube1.obj";
    //const char* fileName = "football.t1.obj";
    //const char* fileName = "football.t1.obj";
    //const char* fileName = "football.t1.obj";


    //mesh = MxMesh_FromFile("/Users/andy/src/mechanica/testing/models/sphere.t1.obj", 1.0, handler);
    //mesh = MxMesh_FromFile("/Users/andy/src/mechanica/testing/models/football.t1.obj", 1.0, handler);
    //mesh = MxMesh_FromFile("/Users/andy/src/mechanica/testing/models/cube1.obj", 1.0, handler);
    mesh = MxMesh_FromFile((dirName + fileName).c_str(), 1.0, &meshObjectTypeHandler);

    // Hook up the cell volume constraints
    VERIFY(propagator->bindConstraint(&cellVolumeConstraint, &redCellType));

    VERIFY(propagator->bindForce(&stdPolygonForce, MxPolygon_Type));

    mesh->selectObject(MxPolygon_Type, 24);

    setTargetVolume(6);
    setTargetVolumeLambda(0.05);


    mesh->setShortCutoff(0);
    mesh->setLongCutoff(0.3);

    pressureMin = 0;
    pressure = 5;
    pressureMax = 15;

}

void GrowthModel::testEdges() {

    return;
}


HRESULT GrowthModel::setStateVector(const float *stateVector)
{
    return S_OK;
}

HRESULT GrowthModel::getStateVector(float *stateVector, uint32_t *count)
{
    *count = 0;
    return S_OK;
}

HRESULT GrowthModel::getStateVectorRate(float time, const float *y, float* dydt)
{
    return S_OK;
}

HRESULT GrowthModel::applyT1Edge2TransitionToSelectedEdge() {
    MxObject *obj = mesh->selectedObject();
    if(obj && dyn_cast<MxEdge>(obj)) {
        return Mx_FlipEdge(mesh, EdgePtr(obj));
    }
    return mx_error(E_FAIL, "no selected object, or selected object is not an edge");
}

HRESULT GrowthModel::applyT2PolygonTransitionToSelectedPolygon()
{
    MxObject *obj = mesh->selectedObject();
    if(obj && dyn_cast<MxPolygon>(obj)) {
        HRESULT result = Mx_CollapsePolygon(mesh, (PolygonPtr)obj);

        if(SUCCEEDED(result)) {

        }

        return result;
    }
    return mx_error(E_FAIL, "no selected object, or selected object is not a polygon");
}

HRESULT GrowthModel::applyT3PolygonTransitionToSelectedPolygon() {
    MxPolygon *poly = dyn_cast<MxPolygon>(mesh->selectedObject());
    if(poly) {

        // make an cut plane perpendicular to the zeroth vertex
        Magnum::Vector3 normal = poly->vertices[0]->position - poly->centroid;

        MxPolygon *p1, *p2;

        HRESULT result = Mx_SplitPolygonBisectPlane(mesh, poly, &normal, &p1, &p2);

        if(SUCCEEDED(result)) {

        }

        return result;
    }
    return mx_error(E_FAIL, "no selected object, or selected object is not a polygon");
}

float GrowthModel::minTargetVolume()
{
    return 0.1 * cellVolumeConstraint.targetVolume;
}

float GrowthModel::maxTargetVolume()
{
    return 2 * cellVolumeConstraint.targetVolume;
}

float GrowthModel::targetVolume()
{
    return cellVolumeConstraint.targetVolume;
}

float GrowthModel::targetVolumeLambda()
{
    return cellVolumeConstraint.lambda;
}

void GrowthModel::setTargetVolumeLambda(float targetVolumeLambda)
{
    cellVolumeConstraint.lambda = targetVolumeLambda;
}

void GrowthModel::setTargetVolume(float tv)
{
    cellVolumeConstraint.targetVolume = tv;
}

float GrowthModel::stdSurfaceTension()
{
    return stdPolygonForce.surfaceTension;
}

void GrowthModel::setStdSurfaceTension(float val)
{
    stdPolygonForce.surfaceTension = val;
}

float GrowthModel::stdSurfaceTensionMin()
{
    return 0;
}

float GrowthModel::stdSurfaceTensionMax()
{
    return stdPolygonForce.surfaceTension * 5;
}

