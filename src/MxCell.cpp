/*
 * MxCell.cpp
 *
 *  Created on: Jul 7, 2017
 *      Author: andy
 */

#include <MxCell.h>
#include <MxMesh.h>
#include <iostream>
#include <algorithm>

#include "MxDebug.h"



static void connectPartialFaces(MxPartialFace &pf1, MxPartialFace &pf2, ushort i1, ushort i2) {
    assert((!is_valid(pf1.neighbors[0]) || !is_valid(pf1.neighbors[1]) || !is_valid(pf1.neighbors[2]))
           && "connecting partial face without empty slots");
    assert((!is_valid(pf2.neighbors[0]) || !is_valid(pf2.neighbors[1]) || !is_valid(pf2.neighbors[2]))
           && "connecting partial face without empty slots");

    for(uint i = 0; i < 3; ++i) {
        assert(pf1.neighbors[i] != i1 && pf1.neighbors[i] != i2);
        if(!is_valid(pf1.neighbors[i])) {
            pf1.neighbors[i] = i2;
            break;
        }
    }

    for(uint i = 0; i < 3; ++i) {
        assert(pf2.neighbors[i] != i1 && pf2.neighbors[i] != i2);
        if(!is_valid(pf2.neighbors[i])) {
            pf2.neighbors[i] = i1;
            break;
        }
    }
}

bool MxCell::connectBoundary(MxMesh& mesh) {
    // clear all of the boundarys before we connect them.
    for(MxPartialFace &pf : boundary) {
        pf.neighbors = invalid<Vector3us>();
    }

    for (uint i = 0; i < boundary.size(); ++i) {
        MxPartialFace &pfi = boundary[i];

        for(uint j = i+1; j < boundary.size(); ++j) {
            MxPartialFace &pfj = boundary[j];

            for(int k = 0; k < 3; ++k) {
                if ((pfi.vertices[0] == pfj.vertices[k] &&
                        (pfi.vertices[1] == pfj.vertices[k+1%3] ||
                         pfi.vertices[1] == pfj.vertices[k+2%3] ||
                         pfi.vertices[2] == pfj.vertices[k+1%3] ||
                         pfi.vertices[2] == pfj.vertices[k+2%3])) ||
                    (pfi.vertices[1] == pfj.vertices[k] &&
                        (pfi.vertices[0] == pfj.vertices[k+1%3] ||
                         pfi.vertices[0] == pfj.vertices[k+2%3] ||
                         pfi.vertices[2] == pfj.vertices[k+1%3] ||
                         pfi.vertices[2] == pfj.vertices[k+2%3])) ||
                    (pfi.vertices[2] == pfj.vertices[k] &&
                        (pfi.vertices[0] == pfj.vertices[k+1%3] ||
                         pfi.vertices[0] == pfj.vertices[k+2%3] ||
                         pfi.vertices[1] == pfj.vertices[k+1%3] ||
                         pfi.vertices[1] == pfj.vertices[k+2%3]))) {



                            std::cout << "face 1" << std::endl;
                            std::cout << "pf[" << i << "] " << pfi.vertices << " {" << std::endl;
                            std::cout << mesh.vertices[pfi.vertices[0]].position << std::endl;
                            std::cout << mesh.vertices[pfi.vertices[1]].position << std::endl;
                            std::cout << mesh.vertices[pfi.vertices[1]].position << std::endl;
                            std::cout << "}" << std::endl;

                            std::cout << "face 2" << std::endl;
                            std::cout << "pf[" << j << "] " << pfj.vertices << " {" << std::endl;
                            std::cout << mesh.vertices[pfj.vertices[0]].position << std::endl;
                            std::cout << mesh.vertices[pfj.vertices[1]].position << std::endl;
                            std::cout << mesh.vertices[pfj.vertices[1]].position << std::endl;
                            std::cout << "}" << std::endl;

                    connectPartialFaces(pfi, pfj, i, j);
                    break;
                }
            }
        }

        // make sure face is completely connected after searching the rest of the
        // set of faces, if not, that means that our surface is not closed.
        if(!is_valid(pfi.neighbors[0]) || !is_valid(pfi.neighbors[1]) || !is_valid(pfi.neighbors[2])) {
            std::cout << "error, surface mesh for cell is not closed" << std::endl;
            return false;
        }
    }

    return true;
}

float MxCell::volume(VolumeMethod vm) {
}

float MxCell::area() {
}

void MxCell::vertexAtributeData(MxMesh& mesh, const std::vector<MxVertexAttribute>& attributes,
        uint vertexCount, uint stride, void* buffer) {
    uchar *ptr = (uchar*)buffer;
    for(uint i = 0; i < boundary.size() && ptr < vertexCount * stride + (uchar*)buffer; ++i, ptr += 3 * stride) {
        //for(auto& attr : attributes) {
        //
        //
        //}
        const MxPartialFace &face = boundary[i];
        Vector3 *ppos = (Vector3*)ptr;
        ppos[0] = mesh.vertices[face.vertices[0]].position;
        ppos[1] = mesh.vertices[face.vertices[1]].position;
        ppos[2] = mesh.vertices[face.vertices[2]].position;
    }
}

void MxCell::indexData(uint indexCount, uint* buffer) {
    for(int i = 0; i < boundary.size(); ++i, buffer += 3) {
        const MxPartialFace &face = boundary[i];
        buffer[0] = face.vertices[0];
        buffer[1] = face.vertices[1];
        buffer[2] = face.vertices[2];
    }
}





void MxCell::dump() {

    for (uint i = 0; i < boundary.size(); ++i) {
        MxPartialFace &pfi = boundary[i];

        std::cout << "face[" << i << "] {" << std::endl;
        std::cout << "vertices:" << pfi.vertices << std::endl;
        std::cout << "}" << std::endl;
    }

}

/* POV mesh format:
mesh2 {
vertex_vectors {
22
,<-1.52898,-1.23515,-0.497254>
,<-2.41157,-0.870689,0.048214>
,<-3.10255,-0.606757,-0.378837>
     ...
,<-2.22371,-1.5823,-2.07175>
,<-2.41157,-1.30442,-2.18785>
}
face_indices {
40
,<1,4,17>
,<1,17,18>
,<1,18,2>
  ...
,<8,21,20>
,<8,20,9>
}
inside_vector <0,0,1>
}
 */

void MxCell::writePOV(std::ostream& out) {
    out << "mesh2 {" << std::endl;
    out << "face_indices {" << std::endl;
    out << boundary.size()  << std::endl;
    for (int i = 0; i < boundary.size(); ++i) {
        out << boundary[i].vertices << std::endl;
        //auto &pf = boundary[i];
        //for (int j = 0; j < 3; ++j) {
            //Vector3& vert = mesh.vertices[pf.vertices[j]].position;
        //        out << ",<" << vert[0] << "," << vert[1] << "," << vert[2] << ">" << std::endl;
        //}
    }
    out << "}" << std::endl;
    out << "}" << std::endl;
}
