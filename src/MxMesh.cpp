/*
 * MxMesh.cpp
 *
 *  Created on: Jan 31, 2017
 *      Author: andy
 */

#include "MxDebug.h"
#include <MxMesh.h>
#include <Magnum/Math/Math.h>


struct Foo {
    Foo() {
        printf("foo");
    }
};

Foo xfoo;

int MxMesh::findVertex(const Magnum::Vector3& pos, double tolerance) {
    for (int i = 0; i < vertices.size(); ++i) {
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

MxCell& MxMesh::createCell() {
    cells.push_back(MxCell{});
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
