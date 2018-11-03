/*
 * CylinderModel.cpp
 *
 * Created on: Sep 20, 2018
 *      Author: andy
 */

#include "CylinderModel.h"
#include <MxDebug.h>
#include <iostream>
#include "MeshIO.h"
#include "CylinderTest.h"
#include "T1Transition.h"
#include "T2Transition.h"
#include "T3Transition.h"
#include "MxCellVolumeConstraint.h"
#include "MxPolygonAreaConstraint.h"
#include <MxPolygonSurfaceTensionForce.h>

MxPolygonType basicPolygonType{"BasicPolygon", MxPolygon_Type};
MxPolygonType growingPolygonType{"GrowingPolygon", MxPolygon_Type};

MxCellVolumeConstraint cellVolumeConstraint{0., 0.};
MxPolygonAreaConstraint areaConstraint{0.1, 0.01};

MxPolygonSurfaceTensionForce stdPolygonForce{0.05};
MxPolygonSurfaceTensionForce growingPolygonForce{0.05};

static struct CylinderCellType : MxCellType
{
    virtual Magnum::Color4 color(struct MxCell *cell) {
        //return Color4{1.0f, 0.0f, 0.0f, 0.08f};
        return Color4::green();
    }

    virtual ~CylinderCellType() {};

    CylinderCellType() : MxCellType{"CylinderCell", MxCell_Type} {};
} cylinderCellType;

static struct MeshObjectTypeHandler : IMeshObjectTypeHandler {
    virtual MxType *cellType(const char* cellName, int cellIndex) {
        return &cylinderCellType;
    }

    virtual MxType *polygonType(int polygonIndex) {
        return &basicPolygonType;
    }

    virtual MxType *partialPolygonType(const MxType *cellType, const MxType *polyType) {
        return nullptr;
    }

    virtual ~MeshObjectTypeHandler() {};

} meshObjectTypeHandler;




CylinderModel::CylinderModel()  {
    growingPolygonType.centerColor = Magnum::Color4::red();
}


HRESULT CylinderModel::loadModel() {
    loadAssImpModel();

    for(int i = 0; i < mesh->cells.size(); ++i) {
        CellPtr cell = mesh->cells[i];
        std::cout << "cell[" << i << "], id:" << cell->id << ", center: " << cell->centroid << std::endl;
    }

    testEdges();

    VERIFY(propagator->structureChanged());

    return S_OK;
}


void CylinderModel::loadAssImpModel() {

    const std::string dirName = "/Users/andy/src/mechanica/testing/models/";

    //const char* fileName = "football.t1.obj";
    //const char* fileName = "football.t2.obj";
    //const char* fileName = "cylinder.1.obj";
    //const char* fileName = "cube1.obj";
    const char* fileName = "hex_cylinder.1.obj";
    //const char* fileName = "football.t1.obj";
    //const char* fileName = "football.t1.obj";

    mesh = MxMesh_FromFile((dirName + fileName).c_str(), 1.0, &meshObjectTypeHandler);

    cellVolumeConstraint.targetVolume = mesh->cells[1]->volume;
    cellVolumeConstraint.lambda = 0.5;

    propagator->bindConstraint(&cellVolumeConstraint, &cylinderCellType);

    propagator->bindForce(&stdPolygonForce, &basicPolygonType);

    propagator->bindForce(&growingPolygonForce, &growingPolygonType);

    mesh->selectObject(MxPolygon_Type, 367);

    CellPtr cell = mesh->cells[1];

    setTargetVolume(cell->volume);
    setTargetVolumeLambda(0.01);

    mesh->setShortCutoff(0);
    mesh->setLongCutoff(0.3);
}

void CylinderModel::testEdges() {
    return;
}

HRESULT CylinderModel::getStateVector(float *stateVector, uint32_t *count)
{
    *count = 0;
    return S_OK;
}

HRESULT CylinderModel::setStateVector(const float *stateVector)
{
    return S_OK;
}


HRESULT CylinderModel::getStateVectorRate(float time, const float *y, float* dydt)
{
    return S_OK;
}

void CylinderModel::setTargetVolume(float tv)
{
    cellVolumeConstraint.targetVolume = tv;
}

HRESULT CylinderModel::applyT1Edge2TransitionToSelectedEdge() {
    MxObject *obj = mesh->selectedObject();
    if(obj && dyn_cast<MxEdge>(obj)) {
        return applyT1Edge2Transition(mesh, EdgePtr(obj));
    }
    return mx_error(E_FAIL, "no selected object, or selected object is not an edge");
}

HRESULT CylinderModel::applyT2PolygonTransitionToSelectedPolygon()
{
    MxObject *obj = mesh->selectedObject();
    if(obj && dyn_cast<MxPolygon>(obj)) {
        HRESULT result = applyT2PolygonTransition(mesh, (PolygonPtr)obj);

        if(SUCCEEDED(result)) {

        }

        return result;
    }
    return mx_error(E_FAIL, "no selected object, or selected object is not a polygon");
}

HRESULT CylinderModel::applyT3PolygonTransitionToSelectedPolygon() {
    MxPolygon *poly = dyn_cast<MxPolygon>(mesh->selectedObject());
    if(poly) {

        // make an cut plane perpendicular to the zeroth vertex
        Magnum::Vector3 normal = poly->vertices[0]->position - poly->centroid;

        MxPolygon *p1, *p2;

        HRESULT result = applyT3PolygonBisectPlaneTransition(mesh, poly, &normal, &p1, &p2);

        if(SUCCEEDED(result)) {

        }
        
        VERIFY(propagator->structureChanged());

        return result;
    }
    return mx_error(E_FAIL, "no selected object, or selected object is not a polygon");
}

float CylinderModel::minTargetVolume()
{
    return 0.1 * cellVolumeConstraint.targetVolume;
}

float CylinderModel::maxTargetVolume()
{
    return 3 * cellVolumeConstraint.targetVolume;
}

float CylinderModel::targetVolume()
{
    return cellVolumeConstraint.targetVolume;
}

float CylinderModel::targetVolumeLambda()
{
    return cellVolumeConstraint.lambda;
}

void CylinderModel::setTargetVolumeLambda(float targetVolumeLambda)
{
    cellVolumeConstraint.lambda = targetVolumeLambda;
}

float CylinderModel::minTargetArea()
{
    return 0.1 * areaConstraint.targetArea;
}

float CylinderModel::maxTargetArea()
{
    return 3 * areaConstraint.targetArea;
}

float CylinderModel::targetArea()
{
    return areaConstraint.targetArea;
}

float CylinderModel::targetAreaLambda()
{
    return areaConstraint.lambda;
}

void CylinderModel::setTargetArea(float targetArea)
{
    areaConstraint.targetArea = targetArea;
}

void CylinderModel::setTargetAreaLambda(float targetAreaLambda)
{
    areaConstraint.lambda = targetAreaLambda;
}

static float PolyDistance = 1;

HRESULT CylinderModel::changePolygonTypes()
{
    MxObject *obj = mesh->selectedObject();
    MxPolygon *poly = dyn_cast<MxPolygon>(obj);

    if(MxType_IsSubtype(obj->ob_type, MxPolygon_Type)) {
        for(PolygonPtr p : mesh->polygons) {
            
            float distance = (poly->centroid - p->centroid).length();
            if(distance <= PolyDistance) {
                VERIFY(MxObject_ChangeType(p, &growingPolygonType));
            }
        }
        VERIFY(propagator->structureChanged());
        return S_OK;
    }
    else {
        return E_FAIL;
    }
}

HRESULT CylinderModel::activateAreaConstraint()
{
    MxObject *obj = mesh->selectedObject();
 
    propagator->bindConstraint(&areaConstraint, &growingPolygonType);
    return propagator->structureChanged();
}

float CylinderModel::stdSurfaceTension()
{
    return stdPolygonForce.surfaceTension;
}

void CylinderModel::setStdSurfaceTension(float val)
{
    stdPolygonForce.surfaceTension = val;
}

float CylinderModel::stdSurfaceTensionMin()
{
    return 0;
}

float CylinderModel::stdSurfaceTensionMax()
{
    return stdPolygonForce.surfaceTension * 5;
}

float CylinderModel::growSurfaceTension()
{
    return growingPolygonForce.surfaceTension;
}

void CylinderModel::setGrowStdSurfaceTension(float val)
{
    growingPolygonForce.surfaceTension = val;
}

float CylinderModel::growSurfaceTensionMin()
{
    return 0;
}

float CylinderModel::growSurfaceTensionMax()
{
    return 5 * growingPolygonForce.surfaceTension;
}
