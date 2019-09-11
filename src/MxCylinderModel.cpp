/*
 * CylinderModel.cpp
 *
 * Created on: Sep 20, 2018
 *      Author: andy
 */

#include "MxCylinderModel.h"
#include <MxDebug.h>
#include <iostream>
#include "MeshIO.h"
#include "MeshOperations.h"
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




MxCylinderModel::MxCylinderModel()  {
    growingPolygonType.centerColor = Magnum::Color4::red();
}


HRESULT MxCylinderModel::loadModel(const char* fileName) {
    loadAssImpModel(fileName);

    for(int i = 0; i < mesh->cells.size(); ++i) {
        CellPtr cell = mesh->cells[i];
        std::cout << "cell[" << i << "], id:" << cell->id << ", center: " << cell->centroid << std::endl;
    }

    testEdges();

    VERIFY(propagator->structureChanged());

    return S_OK;
}


void MxCylinderModel::loadAssImpModel(const char* fileName) {

    std::cout << MX_FUNCTION << ", fileName: " << fileName << std::endl;

    mesh = MxMesh_FromFile(fileName, 1.0, &meshObjectTypeHandler);

    cellVolumeConstraint.targetVolume = mesh->cells[1]->volume;
    cellVolumeConstraint.lambda = 0.5;

    propagator->bindConstraint(&cellVolumeConstraint, &cylinderCellType);

    propagator->bindForce(&stdPolygonForce, &basicPolygonType);

    propagator->bindForce(&growingPolygonForce, &growingPolygonType);

    mesh->selectObject(MxPolygon_Type, 367);

    CellPtr cell = mesh->cells[1];

    setTargetVolume(cell->volume);
    setTargetVolumeLambda(0.01);

    //mesh->setShortCutoff(0);
    //mesh->setLongCutoff(0.3);
}

void MxCylinderModel::testEdges() {
    return;
}

HRESULT MxCylinderModel::getStateVector(float *stateVector, uint32_t *count)
{
    *count = 0;
    return S_OK;
}

HRESULT MxCylinderModel::setStateVector(const float *stateVector)
{
    return S_OK;
}


HRESULT MxCylinderModel::getStateVectorRate(float time, const float *y, float* dydt)
{
    return S_OK;
}

void MxCylinderModel::setTargetVolume(float tv)
{
    cellVolumeConstraint.targetVolume = tv;
}

HRESULT MxCylinderModel::applyT1Edge2TransitionToSelectedEdge() {
    MxObject *obj = mesh->selectedObject();
    if(obj && dyn_cast<MxEdge>(obj)) {
        return Mx_FlipEdge(mesh, EdgePtr(obj));
    }
    return mx_error(E_FAIL, "no selected object, or selected object is not an edge");
}

HRESULT MxCylinderModel::applyT2PolygonTransitionToSelectedPolygon()
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

HRESULT MxCylinderModel::applyT3PolygonTransitionToSelectedPolygon() {
    MxPolygon *poly = dyn_cast<MxPolygon>(mesh->selectedObject());
    if(poly) {

        // make an cut plane perpendicular to the zeroth vertex
        Magnum::Vector3 normal = poly->vertices[0]->position - poly->centroid;

        MxPolygon *p1, *p2;

        HRESULT result = Mx_SplitPolygonBisectPlane(mesh, poly, &normal, &p1, &p2);

        if(SUCCEEDED(result)) {

        }
        
        VERIFY(propagator->structureChanged());

        return result;
    }
    return mx_error(E_FAIL, "no selected object, or selected object is not a polygon");
}

float MxCylinderModel::minTargetVolume()
{
    return 0.1 * cellVolumeConstraint.targetVolume;
}

float MxCylinderModel::maxTargetVolume()
{
    return 3 * cellVolumeConstraint.targetVolume;
}

float MxCylinderModel::targetVolume()
{
    return cellVolumeConstraint.targetVolume;
}

float MxCylinderModel::targetVolumeLambda()
{
    return cellVolumeConstraint.lambda;
}

void MxCylinderModel::setTargetVolumeLambda(float targetVolumeLambda)
{
    cellVolumeConstraint.lambda = targetVolumeLambda;
}

float MxCylinderModel::minTargetArea()
{
    return 0.1 * areaConstraint.targetArea;
}

float MxCylinderModel::maxTargetArea()
{
    return 3 * areaConstraint.targetArea;
}

float MxCylinderModel::targetArea()
{
    return areaConstraint.targetArea;
}

float MxCylinderModel::targetAreaLambda()
{
    return areaConstraint.lambda;
}

void MxCylinderModel::setTargetArea(float targetArea)
{
    areaConstraint.targetArea = targetArea;
}

void MxCylinderModel::setTargetAreaLambda(float targetAreaLambda)
{
    areaConstraint.lambda = targetAreaLambda;
}

static float PolyDistance = 1;

HRESULT MxCylinderModel::changePolygonTypes()
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

HRESULT MxCylinderModel::activateAreaConstraint()
{
    MxObject *obj = mesh->selectedObject();
 
    propagator->bindConstraint(&areaConstraint, &growingPolygonType);
    return propagator->structureChanged();
}

float MxCylinderModel::stdSurfaceTension()
{
    return stdPolygonForce.surfaceTension;
}

void MxCylinderModel::setStdSurfaceTension(float val)
{
    stdPolygonForce.surfaceTension = val;
}

float MxCylinderModel::stdSurfaceTensionMin()
{
    return 0;
}

float MxCylinderModel::stdSurfaceTensionMax()
{
    return stdPolygonForce.surfaceTension * 5;
}

float MxCylinderModel::growSurfaceTension()
{
    return growingPolygonForce.surfaceTension;
}

void MxCylinderModel::setGrowStdSurfaceTension(float val)
{
    growingPolygonForce.surfaceTension = val;
}

float MxCylinderModel::growSurfaceTensionMin()
{
    return 0;
}

float MxCylinderModel::growSurfaceTensionMax()
{
    return 5 * growingPolygonForce.surfaceTension;
}



static void _dealloc(MxCylinderModel *app) {
    std::cout << MX_FUNCTION << std::endl;
}

static PyObject *_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    std::cout << MX_FUNCTION << std::endl;
    Py_RETURN_NONE;
}

static PyMethodDef _methods[] = {
    //{"testImage", (PyCFunction)_testImage, METH_VARARGS,  "make a test image" },
    {NULL}  /* Sentinel */
};

static PyTypeObject _type = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    .tp_name = "mechanica.CylinderModel",
    .tp_basicsize = sizeof(MxCylinderModel),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor)_dealloc,
    .tp_print = 0,
    .tp_getattr = 0,
    .tp_setattr = 0,
    .tp_as_async = 0,
    .tp_repr = 0,
    .tp_as_number = 0,
    .tp_as_sequence = 0,
    .tp_as_mapping = 0,
    .tp_hash = 0,
    .tp_call = 0,
    .tp_str = 0,
    .tp_getattro = 0,
    .tp_setattro = 0,
    .tp_as_buffer = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = 0,
    .tp_traverse = 0,
    .tp_clear = 0,
    .tp_richcompare = 0,
    .tp_weaklistoffset = 0,
    .tp_iter = 0,
    .tp_iternext = 0,
    .tp_methods = _methods,
    .tp_members = 0,
    .tp_getset = 0,
    .tp_base = 0,
    .tp_dict = 0,
    .tp_descr_get = 0,
    .tp_descr_set = 0,
    .tp_dictoffset = 0,
    .tp_init = 0,
    .tp_alloc = 0,
    .tp_new = _new,
    .tp_free = 0,
    .tp_is_gc = 0,
    .tp_bases = 0,
    .tp_mro = 0,
    .tp_cache = 0,
    .tp_subclasses = 0,
    .tp_weaklist = 0,
    .tp_del = 0,
    .tp_version_tag = 0,
    .tp_finalize = 0,
};

PyTypeObject *MxCylinderModel_Type = &_type;

HRESULT MxCylinderModel_init(PyObject* m) {

    std::cout << MX_FUNCTION << std::endl;


    if (PyType_Ready((PyTypeObject *)MxCylinderModel_Type) < 0)
        return E_FAIL;


    Py_INCREF(MxCylinderModel_Type);
    PyModule_AddObject(m, "CylinderModel", (PyObject *) MxCylinderModel_Type);

    return 0;
}
