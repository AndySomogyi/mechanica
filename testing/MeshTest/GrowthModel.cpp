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

    mesh->selectObject(MxPolygon_Type, 24);
    
    setTargetVolume(6);
    setTargetVolumeLambda(0.05);


    mesh->setShortCutoff(0);
    mesh->setLongCutoff(0.3);

    pressureMin = 0;
    pressure = 5;
    pressureMax = 15;

    cellMediaSurfaceTensionMin = 0;
    cellMediaSurfaceTension = 0.1;
    cellMediaSurfaceTensionMax = 1;

    cellCellSurfaceTensionMin = 0;
    cellCellSurfaceTension = 0.1;
    cellCellSurfaceTensionMax = 1;
}




HRESULT GrowthModel::calcForce() {

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

void GrowthModel::applyDifferentialSurfaceTension() {

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

            /*

            for(int v = 0; v < 3; ++v) {
                Vector3 p1 = tri->vertices[(v+1)%3]->position;
                Vector3 p2 = tri->vertices[(v+2)%3]->position;
                float len = (p1-p2).length();

                tri->partialTriangles[0].force[v]
                    += differentialSurfaceTension * len * (tri->vertices[v]->position - tri->centroid).normalized();

                tri->partialTriangles[1].force[v]
                    += differentialSurfaceTension * len * (tri->vertices[v]->position - tri->centroid).normalized();
            }
            */
        }
        else {
            //std::cout << "continuing...";
        }
    }
}

HRESULT GrowthModel::cellAreaForce(CellPtr cell) {

    //return S_OK;

    if(mesh->rootCell() == cell) {
        return S_OK;
    }

    assert(cell->area >= 0);

    float diff =  - cell->area;
    //float diff = -0.35;

    for(auto pt: cell->surface) {

        PolygonPtr tri = pt->polygon;

        assert(tri->area >= 0);

        //float areaFraction = tri->area / cell->area;

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
            //pt->force[v] -= surfaceTension * (tri->vertices[v]->position - tri->centroid).normalized();

            Vector3 p1 = tri->vertices[(v+1)%3]->position;
            Vector3 p2 = tri->vertices[(v+2)%3]->position;
            float len = (p1-p2).length();
            //pt->force[v] -= surfaceTension * len * (tri->vertices[v]->position - tri->centroid).normalized();
            //pt->force[v] -= surfaceTension * (tri->vertices[v]->position - tri->centroid);

            //for(int i = 0; i < 3; ++i) {
            //    if(i != v) {
            //        pt->force[v] -= 0.5 * (tri->vertices[v]->position - tri->vertices[i]->position);
            //    }
            //}
            //pt->force[v] +=  -100 * areaFraction * dir[v] / totLen;
            //pt->force[v] +=  10.5 * diff * (tri->area / cell->area) *  dir[v].normalized();
            // pt->force[v] +=  -30.5 * (pt->triangle->vertices[v]->position - pt->triangle->centroid);
            //pt->force[v] += -300 * areaFraction * (pt->triangle->vertices[v]->position - pt->triangle->centroid).normalized();

            //for(int o = 0; o < 3; ++o) {
            //    if(o != v) {
            //        pt->force[v] +=  -10.5 * (pt->triangle->vertices[v]->position - pt->triangle->vertices[o]->position);
            //    }
            //}
        }

    }
    return S_OK;
}

void GrowthModel::testEdges() {

    return;
}


HRESULT GrowthModel::getForces(float time, uint32_t len, const Vector3* pos, Vector3* force)
{
    HRESULT result;



    return S_OK;
}

HRESULT GrowthModel::getMasses(float time, uint32_t len, float* masses)
{
    if(len != mesh->vertices.size()) {
        return E_FAIL;
    }

    for(int i = 0; i < len; ++i) {
        masses[i] = mesh->vertices[i]->mass;
    }
    return S_OK;
}

HRESULT GrowthModel::getPositions(float time, uint32_t len, Vector3* pos)
{
    for(int i = 0; i < len; ++i) {
        pos[i] = mesh->vertices[i]->position;
    }
    return S_OK;
}

HRESULT GrowthModel::setPositions(float time, uint32_t len, const Vector3* pos)
{
    return mesh->setPositions(len, pos);
}

HRESULT GrowthModel::getAccelerations(float time, uint32_t len,
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

HRESULT GrowthModel::getStateVector(float *stateVector, uint32_t *count)
{
    *count = 0;
    return S_OK;
}

HRESULT GrowthModel::setStateVector(const float *stateVector)
{
    return S_OK;
}


HRESULT GrowthModel::getStateVectorRate(float time, const float *y, float* dydt)
{
    return S_OK;
}

HRESULT GrowthModel::applySurfaceTensionForce(PolygonPtr pp) {
    float k = 0;

    if(pp->cells[0]->isRoot() || pp->cells[1]->isRoot()) {
        k = cellMediaSurfaceTension;
    } else {
        k = cellCellSurfaceTension;
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

HRESULT GrowthModel::applyT1Edge2TransitionToSelectedEdge() {
    MxObject *obj = mesh->selectedObject();
    if(obj && dyn_cast<MxEdge>(obj)) {
        return applyT1Edge2Transition(mesh, EdgePtr(obj));
    }
    return mx_error(E_FAIL, "no selected object, or selected object is not an edge");
}

HRESULT GrowthModel::applyT2PolygonTransitionToSelectedPolygon()
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

HRESULT GrowthModel::applyT3PolygonTransitionToSelectedPolygon() {
    MxPolygon *poly = dyn_cast<MxPolygon>(mesh->selectedObject());
    if(poly) {
        
        // make an cut plane perpendicular to the zeroth vertex
        Magnum::Vector3 normal = poly->vertices[0]->position - poly->centroid;
        
        MxPolygon *p1, *p2;
        
        HRESULT result = applyT3PolygonBisectPlaneTransition(mesh, poly, &normal, &p1, &p2);
 
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
    
#ifndef NEW_CONSTRAINTS
    for(CellPtr cell : mesh->cells) {
        cell->targetVolume = tv;
    }
#endif
}
