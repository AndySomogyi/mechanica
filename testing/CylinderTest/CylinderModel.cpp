/*
 * GrowthModel.cpp
 *
 *  Created on: Oct 13, 2017
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

MxPolygonType basicPolygonType{"BasicPolygon", MxPolygon_Type};
MxPolygonType growingPolygonType{"GrowingPolygon", MxPolygon_Type};

MxCellVolumeConstraint cellVolumeConstraint{0., 0.};
MxPolygonAreaConstraint areaConstraint{0.1, 0.001};

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

    mesh->selectObject(MxPolygon_Type, 367);

    CellPtr cell = mesh->cells[1];

    setTargetVolume(cell->volume);
    setTargetVolumeLambda(0.01);

    mesh->setShortCutoff(0);
    mesh->setLongCutoff(0.3);

    cellMediaSurfaceTensionMin = 0;
    cellMediaSurfaceTension = 0.05;
    cellMediaSurfaceTensionMax = 3.0;
}




HRESULT CylinderModel::calcForce() {

    HRESULT result;

    /*

    for(PolygonPtr poly : mesh->polygons) {
        applyVolumeConservationForce(poly->cells[0], poly, &poly->partialPolygons[0]);
        applyVolumeConservationForce(poly->cells[1], poly, &poly->partialPolygons[1]);
    }
    */

    for(PolygonPtr poly : mesh->polygons) {
        applySurfaceTensionForce(poly);
    }



    //applyDifferentialSurfaceTension();

    //centerOfMassForce(mesh->cells[1], mesh->cells[3], harmonicBondStrength);

    //centerOfMassForce(mesh->cells[22], mesh->cells[7], 1);

    return S_OK;
}

void CylinderModel::applyDifferentialSurfaceTension() {

    for(PolygonPtr tri : mesh->polygons) {

        if((tri->cells[0] == mesh->cells[1] && tri->cells[1] == mesh->cells[2]) ||
           (tri->cells[0] == mesh->cells[2] && tri->cells[1] == mesh->cells[1])) {

            assert(tri->area >= 0);

            Vector3 dir[3];
            float len[3];
            float totLen = 0;
            for(int v = 0; v < 3; ++v) {
                dir[v] = tri->vertices[v]->position - tri->centroid;
                len[v] = dir[v].length();
                totLen += len[v];
            }
        }
        else {
            //std::cout << "continuing...";
        }
    }
}

void CylinderModel::testEdges() {
    return;
}

HRESULT CylinderModel::getForces(float time, uint32_t len, const Vector3* pos, Vector3* force)
{
    HRESULT result;
    return S_OK;
}

HRESULT CylinderModel::getMasses(float time, uint32_t len, float* masses)
{
    if(len != mesh->vertices.size()) {
        return E_FAIL;
    }

    for(int i = 0; i < len; ++i) {
        masses[i] = mesh->vertices[i]->mass;
    }
    return S_OK;
}

HRESULT CylinderModel::getPositions(float time, uint32_t len, Vector3* pos)
{
    for(int i = 0; i < len; ++i) {
        pos[i] = mesh->vertices[i]->position;
    }
    return S_OK;
}

HRESULT CylinderModel::setPositions(float time, uint32_t len, const Vector3* pos)
{
    return mesh->setPositions(len, pos);
}

HRESULT CylinderModel::getAccelerations(float time, uint32_t len,
        const Vector3* pos, Vector3* acc)
{
    HRESULT result;

    if(len != mesh->vertices.size()) {
        return E_FAIL;
    }

    if(pos) {
        if(!SUCCEEDED(result = mesh->setPositions(len, pos))) {
            return result;
        }
    }

    calcForce();

    for(int i = 0; i < mesh->vertices.size(); ++i) {
        VertexPtr v = mesh->vertices[i];

        acc[i] = v->force;
    }

    return S_OK;
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

HRESULT CylinderModel::applySurfaceTensionForce(PolygonPtr pp) {
    float k = 0;

    if(pp->cells[0]->isRoot() || pp->cells[1]->isRoot()) {
        k = cellMediaSurfaceTension;
    } else {
        k = 0;
    }

    for(uint i = 0; i < pp->vertices.size(); ++i) {
        VertexPtr vi = pp->vertices[i];
        VertexPtr vn = pp->vertices[(i+1)%pp->vertices.size()];
        Vector3 dx = vn->position - vi->position;

        vi->force += k * dx;
        vn->force -= k * dx;
    }

    return S_OK;
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

HRESULT CylinderModel::changePolygonTypes()
{
    MxObject *obj = mesh->selectedObject();
    MxPolygon *poly = dyn_cast<MxPolygon>(obj);

    if(MxType_IsSubtype(obj->ob_type, MxPolygon_Type)) {
        VERIFY(MxObject_ChangeType(obj, &growingPolygonType));
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
