/*
 * MxMesh.cpp
 *
 *  Created on: Jan 31, 2017
 *      Author: andy
 */

#include "MxDebug.h"
#include <MxMesh.h>
#include <Magnum/Math/Math.h>


int MxMesh::findVertex(const Magnum::Vector3& pos, double tolerance) {
    for (int i = 1; i < vertices.size(); ++i) {
        float dist = (vertices[i].position - pos).dot();
        if (dist <= tolerance) {
            return i;
        }
    }
    return -1;
}

uint MxMesh::addVertex(const Magnum::Vector3& pos) {
    vertices.push_back({pos, {}, {}});
    return vertices.size() - 1;
}

MxCell& MxMesh::createCell(MxCellType *type) {
    cells.push_back(MxCell{type, this, nullptr});
    return cells[cells.size() - 1];
}

void MxMesh::vertexAtributes(const std::vector<MxVertexAttribute>& attributes,
        uint vertexCount, uint stride, void* buffer) {
}


void MxMesh::dump(uint what) {
    for(int i = 0; i < vertices.size(); ++i) {
        std::cout << "[" << i << "]" << vertices[i].position << std::endl;
    }
}

#include <random>

std::default_random_engine eng;

void MxMesh::jiggle() {

    std::uniform_real_distribution<float> distribution(-0.002,0.002);

    for (int i = 0; i < vertices.size(); ++i) {

        Vector3 test = vertices[i].position + Vector3{distribution(eng), distribution(eng), distribution(eng)};

        if((test - initPos[i]).length() < 0.7) {
            vertices[i].position  = test;
        }
    }
}

std::tuple<Magnum::Vector3, Magnum::Vector3> MxMesh::extents() {

    auto min = Vector3{std::numeric_limits<float>::max()};
    auto max = Vector3{std::numeric_limits<float>::min()};


    for(auto& v : vertices) {
        for(int i = 0; i < 3; ++i) {min[i] = (v.position[i] < min[i] ? v.position[i] : min[i]);}
        for(int i = 0; i < 3; ++i) {max[i] = (v.position[i] > max[i] ? v.position[i] : max[i]);}
    }

    return std::make_tuple(min, max);
}

TriangleIndx MxMesh::createTriangle(const std::array<VertexIndx, 3> &vertInd) {
    for (MxTriangle& tri : triangles) {
        if (tri.matchVertexIndices(vertInd) != 0) {
            return &tri - &triangles[0];
        }
    }

    triangles.push_back(MxTriangle{vertInd});
    return triangles.size() - 1;
}


MxPartialTriangle& MxMesh::createPartialTriangle(
        MxPartialTriangleType* type, MxCell& cell,
        TriangleIndx triIndx, const PTriangleIndices& neighbors)
{
    MxTriangle &t = triangle(triIndx);
    if(t.cells[0] == 0 && t.partialTriangles[0] == 0)
    {
        t.cells[0] = cellId(cell);
        partialTriangles.push_back(MxPartialTriangle{type, triIndx, neighbors, 0.0, nullptr});
        t.partialTriangles[0] = partialTriangles.size() - 1;
        cell.boundary.push_back(t.partialTriangles[0]);
        return partialTriangles.back();
    }
    else if(t.cells[1] == 0 && t.partialTriangles[1] == 0)
    {
        t.cells[1] = cellId(cell);
        partialTriangles.push_back(MxPartialTriangle{type, triIndx, neighbors, 0.0, nullptr});
        t.partialTriangles[1] = partialTriangles.size() - 1;
        cell.boundary.push_back(t.partialTriangles[1]);
        return partialTriangles.back();
    }
    else {
        assert(0 && "invalid triangle");
        throw(0);
    }
}

MxPartialTriangle& MxMesh::createPartialTriangle(
        MxPartialTriangleType* type, MxCell& cell,
        const VertexIndices& vertIndices, const PTriangleIndices& neighbors)
{
    TriangleIndx ti = createTriangle(vertIndices);
    return createPartialTriangle(type, cell, ti, neighbors);
}

MxCellType universeCellType = {};

MxCellType *MxUniverseCell_Type = &universeCellType;

MxPartialTriangleType universePartialTriangleType = {};

MxPartialTriangleType *MxUniversePartialTriangle_Type =
        &universePartialTriangleType;

MxMesh::MxMesh()
{
    createCell(MxUniverseCell_Type);
    vertices.push_back(MxMeshVertex{});
    triangles.push_back(MxTriangle{{{0,0,0}}});
    partialTriangles.push_back(MxPartialTriangle{MxUniversePartialTriangle_Type, 0, {{0,0,0}}, 0.0, nullptr});
}
