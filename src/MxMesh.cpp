/*
 * MxMesh.cpp
 *
 *  Created on: Jan 31, 2017
 *      Author: andy
 */


#include "MxDebug.h"
#include <MxMesh.h>
#include <Magnum/Math/Math.h>
#include "MagnumExternal/Optional/optional.hpp"

#include <deque>
#include <limits>
#include <cmath>

#define MESHOP

int MxMesh::findVertex(const Magnum::Vector3& pos, double tolerance) {
    for (int i = 1; i < vertices.size(); ++i) {
        float dist = (vertices[i]->position - pos).dot();
        if (dist <= tolerance) {
            return i;
        }
    }
    return -1;
}

VertexPtr MxMesh::createVertex(const Magnum::Vector3& pos) {
    VertexPtr v = new MxVertex{0., 0., pos};
    vertices.push_back(v);
    assert(valid(v));
    return v;
}

CellPtr MxMesh::createCell(MxCellType *type) {
    CellPtr cell = new MxCell{type, this, nullptr};
    cells.push_back(cell);
    cell->ob_refcnt = cells.size() - 1;
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

TrianglePtr MxMesh::findTriangle(const std::array<VertexPtr, 3> &verts) {
    assert(valid(verts[0]));
    assert(valid(verts[1]));
    assert(valid(verts[2]));

    for (TrianglePtr tri : triangles) {
        if (tri->matchVertexIndices(verts) != 0) {
            return tri;
        }
    }

    return nullptr;
}






MxCellType universeCellType = {};

MxCellType *MxUniverseCell_Type = &universeCellType;

MxPartialTriangleType universePartialTriangleType = {};

MxPartialTriangleType *MxUniversePartialTriangle_Type =
        &universePartialTriangleType;

MxMesh::MxMesh() :
        meshOperations(this, 0, 1.5)
{
    _rootCell = createCell();
}

FacetPtr MxMesh::findFacet(CellPtr a, CellPtr b) {
    for(auto facet : a->facets) {
        if (incident(facet, b)) {
            return facet;
        }
    }
    return nullptr;
}

FacetPtr MxMesh::createFacet(MxFacetType* type) {
    FacetPtr facet = new MxFacet{type, this, {{nullptr, nullptr}}};
    facets.push_back(facet);
    return facet;
}

FacetPtr MxMesh::findFacet(const std::array<VertexPtr, 4>& verts) {
    for(FacetPtr facet : facets) {
        if(facet->triangles.size() != 2) continue;

        // each triangle has to be incident to two vertices.

        bool inc = true;
        for(TrianglePtr tri : facet->triangles) {
            int incCnt = 0;
            for(VertexPtr vert : verts) {
                if(incident(tri, vert)) {
                    incCnt += 1;
                }
            }
            inc &= (incCnt == 3);
        }
        if(inc) return facet;
    }
    return nullptr;
}

HRESULT MxMesh::deleteVertex(VertexPtr v) {
#ifdef MESHOP
    meshOperations.removeDependentOperations(v);
#else
    longEdges.remove(v);
    shortEdges.remove(v);
    validateEnquedEdges();
#endif
    remove(vertices, v);
#ifndef NDEBUG
    for(TrianglePtr tri : triangles) {
        assert(!incident(tri, v));
    }
#endif
    delete v;
    return S_OK;
}

HRESULT MxMesh::deleteTriangle(TrianglePtr tri) {

#ifdef MESHOP
    meshOperations.removeDependentOperations(tri);
#else
    longEdges.remove(tri);
    shortEdges.remove(tri);
    validateEnquedEdges();
#endif


    remove(triangles, tri);
    delete tri;

    assert(!contains(triangles, tri));

#ifndef NDEBUG
    for(CellPtr cell : cells) {
        assert(!incident(cell, tri));
    }
    for(FacetPtr facet : facets) {
        assert(!contains(facet->triangles, tri));
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



bool MxMesh::valid(TrianglePtr p) {

    if(std::find(triangles.begin(), triangles.end(), p) == triangles.end()) {
        return false;
    }


    return
        p == p->partialTriangles[0].triangle &&
        p == p->partialTriangles[1].triangle &&
        valid(p->vertices[0]) &&
        valid(p->vertices[1]) &&
        valid(p->vertices[2]);
}

bool MxMesh::valid(CellPtr c) {
    if(std::find(cells.begin(), cells.end(), c) == cells.end()) {
        return false;
    }

    for(PTrianglePtr p : c->boundary) {
        if(!valid(p)) {
            return false;
        }
    }

    return true;
}

bool MxMesh::valid(VertexPtr v) {
    return std::find(vertices.begin(), vertices.end(), v) != vertices.end();
}

bool MxMesh::valid(PTrianglePtr p) {
    return p && valid(p->triangle);
}

MxMesh::~MxMesh() {
    for(auto c : cells) {
        delete c;
    }
    for(auto p : vertices) {
        delete p;
    }
    for(auto t : triangles) {
        delete t;
    }
}

HRESULT MxMesh::positionsChanged() {
    HRESULT result = E_FAIL;

    for(VertexPtr v : vertices) {
        v->mass = 0;
        v->area = 0;
    }

    for(TrianglePtr tri : triangles) {
        if((result = tri->positionsChanged() != S_OK)) {
            return result;
        }

#ifndef MESHOP

        // TODO: should we have explicit edges??? Save on compute time.
        for(int i = 0; i < 3; ++i) {
            float d = (tri->vertices[i]->position - tri->vertices[(i+1)%3]->position).length();

            if (d <= meshOperations.getShortCutoff()) {
                enqueueShortEdge(tri->vertices[i], tri->vertices[(i+1)%3]);
            }

            if (d >= meshOperations.getLongCutoff()) {
                enqueueLongEdge(tri->vertices[i], tri->vertices[(i+1)%3]);
            }
        }
#endif
    }

#ifdef MESHOP
    if((result = meshOperations.positionsChanged(triangles.begin(), triangles.end())) != S_OK) {
        return result;
    }
    if((result = meshOperations.apply()) != S_OK) {
        return result;
    }
#else
    if((result = processOffendingEdges()) != S_OK) {
        return result;
    }
#endif

    for(FacetPtr facet : facets) {
        if((result = facet->positionsChanged() != S_OK)) {
            return result;
        }
    }

    for(CellPtr cell : cells) {
        if((result = cell->positionsChanged() != S_OK)) {
            return result;
        }
    }

    return S_OK;
}

TrianglePtr MxMesh::createTriangle(MxTriangleType* type,
        const std::array<VertexPtr, 3>& verts) {

    TrianglePtr tri = new MxTriangle{type, verts};
    triangles.push_back(tri);

    assert(valid(tri));

    return tri;
}

bool MxMesh::validateVertex(const VertexPtr v) {
    assert(contains(vertices, v));
    for(TrianglePtr tri : v->triangles()) {
        assert(incident(tri, v));
        assert(contains(triangles, tri));
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
    for(int i = 0; i < triangles.size(); ++i) {
        validateTriangle(triangles[i]);
    }
    return true;
}

bool MxMesh::validateTriangle(const TrianglePtr tri) {
    assert(tri->isValid());

    for(int i = 0; i < 3; ++i) {
        validateVertex(tri->vertices[i]);
        assert(contains(tri->vertices[i]->triangles(), tri));
    }
    return true;
}

bool MxMesh::validate() {
    return true;
    validateTriangles();
    validateVertices();
    return true;
}

bool MxMesh::validateEdge(const VertexPtr a, const VertexPtr b) {
    MxEdge e(a, b);
    return true;
}

