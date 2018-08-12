/*
 * MxMesh.cpp
 *
 *  Created on: Jan 31, 2017
 *      Author: andy
 */


#include "MxDebug.h"
#include "MxMesh.h"
#include <Magnum/Math/Math.h>
#include "MagnumExternal/Optional/optional.hpp"

#include <deque>
#include <limits>
#include <cmath>


int MxMesh::findVertex(const Magnum::Vector3& pos, double tolerance) {
    for (int i = 1; i < vertices.size(); ++i) {
        float dist = (vertices[i]->position - pos).dot();
        if (dist <= tolerance) {
            return i;
        }
    }
    return -1;
}

VertexPtr MxMesh::createVertex(const Magnum::Vector3& pos, const MxType *type) {

    VertexPtr retval = nullptr;
    retval = new MxVertex{0., 0., pos};


    retval->id = ++vertexId;
    vertices.push_back(retval);
    return retval;
}


MxObject *MxMesh::alloc(const MxType* type)
{
    VertexPtr retval = nullptr;
    if(type == MxVertex_Type) {
        retval = new MxVertex();
        vertices.push_back(retval);
        return retval;
    }
    else if(type == MxSkeletalEdge_Type) {
        MxEdge *e = new MxEdge();
        edges.push_back(e);
        return e;
    }
    else {
        assert(0);
        return nullptr;
    }
}

CellPtr MxMesh::createCell(MxCellType *type) {
    CellPtr cell = new MxCell{(uint)cells.size(), type, this, nullptr};
    cells.push_back(cell);
    return cell;
}

void MxMesh::vertexAtributes(const std::vector<MxVertexAttribute>& attributes,
        uint vertexCount, uint stride, void* buffer) {
}


void MxMesh::dump(uint what) {
    for(int i = 0; i < vertices.size(); ++i) {
        std::cout << "[" << i << "]" << vertices[i]->position << std::endl;
    }
}

std::tuple<Magnum::Vector3, Magnum::Vector3> MxMesh::extents() {

    auto min = Vector3{std::numeric_limits<float>::max()};
    auto max = Vector3{std::numeric_limits<float>::min()};


    for(auto& v : vertices) {
        for(int i = 0; i < 3; ++i) {min[i] = (v->position[i] < min[i] ? v->position[i] : min[i]);}
        for(int i = 0; i < 3; ++i) {max[i] = (v->position[i] > max[i] ? v->position[i] : max[i]);}
    }

    return std::make_tuple(min, max);
}

struct UniverseCellType : MxCellType {
    virtual Magnum::Color4 color(struct MxCell *cell) {
        return Color4{0., 0., 0., 0.};
    }
};

UniverseCellType universeCellType = {};

MxCellType *MxUniverseCell_Type = &universeCellType;

MxPartialPolygonType universePartialTriangleType = {};

MxPartialPolygonType *MxUniversePartialTriangle_Type =
        &universePartialTriangleType;

MxMesh::MxMesh() :
        meshOperations(this, 0, 1.5)
{
    _rootCell = createCell();
}



HRESULT MxMesh::deleteVertex(VertexPtr v) {

    meshOperations.removeDependentOperations(v);

    remove(vertices, v);
#ifndef NDEBUG
    for(PolygonPtr tri : polygons) {
        assert(!incidentPolygonVertex(tri, v));
    }
#endif
    delete v;
    return S_OK;
}

HRESULT MxMesh::deletePolygon(PolygonPtr tri) {

    meshOperations.removeDependentOperations(tri);

    remove(polygons, tri);
    delete tri;

    assert(!contains(polygons, tri));

#ifndef NDEBUG
    for(CellPtr cell : cells) {
        assert(!connectedCellPolygonPointers(cell, tri));
    }

#endif
    return S_OK;
}


int test(const std::vector<std::string*> &stuff) {

    for(int i = 0; i < stuff.size(); ++i) {
        std::string *s = stuff[i];

        s->append("foo");
    }

    //stuff.push_back("");
    return 5;
}



bool MxMesh::valid(PolygonPtr p) {

    if(std::find(polygons.begin(), polygons.end(), p) == polygons.end()) {
        return false;
    }


    return
        p == p->partialTriangles[0].polygon &&
        p == p->partialTriangles[1].polygon &&
        valid(p->vertices[0]) &&
        valid(p->vertices[1]) &&
        valid(p->vertices[2]);
}

bool MxMesh::valid(CellPtr c) {
    if(std::find(cells.begin(), cells.end(), c) == cells.end()) {
        return false;
    }

    for(PPolygonPtr p : c->boundary) {
        if(!valid(p)) {
            return false;
        }
    }

    return true;
}

bool MxMesh::valid(VertexPtr v) {
    return std::find(vertices.begin(), vertices.end(), v) != vertices.end();
}

bool MxMesh::valid(PPolygonPtr p) {
    return p && valid(p->polygon);
}

MxMesh::~MxMesh() {
    for(auto c : cells) {
        delete c;
    }
    for(auto p : vertices) {
        delete p;
    }
    for(auto t : polygons) {
        delete t;
    }
}

HRESULT MxMesh::applyMeshOperations() {
    HRESULT result = E_FAIL;

    MxVertex::maxForceDivergence = std::numeric_limits<float>::min();
    MxVertex::minForceDivergence = std::numeric_limits<float>::max();

    if((result = meshOperations.positionsChanged(polygons.begin(), polygons.end())) != S_OK) {
        return result;
    }

    for(VertexPtr vert : vertices) {
        meshOperations.valenceChanged(vert);
    }

    if((result = meshOperations.apply()) != S_OK) {
        return result;
    }

    return updateDerivedAttributes();
}

PolygonPtr MxMesh::createPolygon(MxPolygonType* type,
        const std::vector<VertexPtr>& verts) {

    PolygonPtr tri = new MxPolygon{(uint)polygons.size(), type, verts};

    polygons.push_back(tri);

    //assert(tri->isValid());

    return tri;
}

bool MxMesh::validateVertex(const VertexPtr v) {
    assert(contains(vertices, v));
    for(PolygonPtr tri : v->triangles()) {
        assert(incidentPolygonVertex(tri, v));
        assert(contains(polygons, tri));
        assert(tri->cells[0] && tri->cells[1]);
    }
    return true;
}

bool MxMesh::validateVertices() {
    for(int i = 0; i < vertices.size(); ++i) {
        validateVertex(vertices[i]);
    }
    return true;
}

bool MxMesh::validateTriangles() {
    bool result = true;
    for(int i = 0; i < polygons.size(); ++i) {
        PolygonPtr tri = polygons[i];
        result &= tri->isValid();
    }
    return true;
}

bool MxMesh::validateTriangle(const PolygonPtr tri) {
    assert(tri->isValid());

    for(int i = 0; i < 3; ++i) {
        validateVertex(tri->vertices[i]);
        assert(contains(tri->vertices[i]->triangles(), tri));
    }
    return true;
}

bool MxMesh::validate() {

    validateTriangles();

    return true;
    validateVertices();
    return true;
}

HRESULT MxMesh::updateDerivedAttributes()
{
    HRESULT result;

    MxVertex::maxForceDivergence = std::numeric_limits<float>::min();
    MxVertex::minForceDivergence = std::numeric_limits<float>::max();

    for(int i = 0; i < vertices.size(); ++i) {
        VertexPtr v = vertices[i];
        v->mass = 0;
        v->area = 0;
    }

    for(PolygonPtr tri : polygons) {

        tri->partialTriangles[0].force[0] = Vector3{};
        tri->partialTriangles[0].force[1] = Vector3{};
        tri->partialTriangles[0].force[2] = Vector3{};
        tri->partialTriangles[1].force[0] = Vector3{};
        tri->partialTriangles[1].force[1] = Vector3{};
        tri->partialTriangles[1].force[2] = Vector3{};

        if((result = tri->positionsChanged() != S_OK)) {
            return result;
        }
    }

    for(CellPtr cell : cells) {
        if((result = cell->updateDerivedAttributes() != S_OK)) {
            return result;
        }
    }

    return S_OK;
}

SkeletalEdgePtr MxMesh::findSkeletalEdge(CVertexPtr a, CVertexPtr b) const
{
    for(SkeletalEdgePtr edge : edges) {
        if(edge->matches(a, b)) {
            return edge;
        }
    }
    return nullptr;
}



void MxMesh::markEdge(const Edge& edge) {
    for(PolygonPtr tri : polygons) {
        tri->color = Magnum::Color4{0.0f, 0.0f, 0.0f, 0.0f};
    }
}

void MxMesh::markTriangle(const PolygonPtr tri) {
    makeTrianglesTransparent();
    tri->color = Magnum::Color4::red();
}

HRESULT MxMesh::valenceChanged(VertexPtr v)
{
    return S_OK;
}


HRESULT MxMesh::setPositions(uint32_t len, const Vector3* positions)
{
    HRESULT result;

    MxVertex::maxForceDivergence = std::numeric_limits<float>::min();
    MxVertex::minForceDivergence = std::numeric_limits<float>::max();

    if(positions) {
        for(int i = 0; i < vertices.size(); ++i) {
            VertexPtr v = vertices[i];
            v->mass = 0;
            v->area = 0;
            v->position = positions[i];
        }
    }
    else {
        for(int i = 0; i < vertices.size(); ++i) {
            VertexPtr v = vertices[i];
            v->mass = 0;
            v->area = 0;
        }
    }

    for(PolygonPtr tri : polygons) {

        tri->partialTriangles[0].force[0] = Vector3{};
        tri->partialTriangles[0].force[1] = Vector3{};
        tri->partialTriangles[0].force[2] = Vector3{};
        tri->partialTriangles[1].force[0] = Vector3{};
        tri->partialTriangles[1].force[1] = Vector3{};
        tri->partialTriangles[1].force[2] = Vector3{};

        if((result = tri->positionsChanged() != S_OK)) {
            return result;
        }
    }

    for(CellPtr cell : cells) {
        if((result = cell->updateDerivedAttributes() != S_OK)) {
            return result;
        }
    }

    return S_OK;
}


