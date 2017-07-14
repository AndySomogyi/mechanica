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

uint MxMesh::appendVertex(const Magnum::Vector3& pos, double tolerance) {
    int i = findVertex(pos, tolerance);
    if (i >= 0) {
        return i;
    } else {
        vertices.push_back({pos, {}, {}});
        return vertices.size() - 1;
    }
}

MxCell& MxMesh::createCell() {
    cells.push_back(MxCell{*this});
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



