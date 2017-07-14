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

#include <random>

std::default_random_engine eng;

void MxMesh::jiggle() {
    
    
    //std::uniform_real_distribution<double> distribution(-0.1,0.1);
    
    std::normal_distribution<float> distribution(0.0,0.1);
    
    for (int i = 0; i < vertices.size(); ++i) {
        
        Vector3 test = vertices[i].position + Vector3{distribution(eng), distribution(eng), distribution(eng)};
        
   
        if((test - initPos[i]).length() < 2) {
            vertices[i].position  = test;
        }
        
    }
    
 
}



