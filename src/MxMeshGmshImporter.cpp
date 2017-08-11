/*
 * MxMeshGmshImporter.cpp
 *
 *  Created on: Jul 18, 2017
 *      Author: andy
 */

#include <iostream>
#include <MxMeshGmshImporter.h>


#include <Magnum/Math/Vector3.h>


using Gmsh::ElementType;

static inline Magnum::Vector3 makeVertex(const double pos[]) {
    return Vector3{float(pos[0]), float(pos[1]), float(pos[2])};
}



bool MxMeshGmshImporter::read(const std::string& path) {

    gmsh = Gmsh::Mesh::read(path);

    for(auto& element : gmsh.elements) {
        switch(element.type) {
        case ElementType::Hexahedron: {
            const Gmsh::Hexahedron &h = element.get<Gmsh::Hexahedron>();
            addCell(h);
            break;
        }
        default: continue;
        }
    }

    mesh.initPos.resize(mesh.vertices.size());

    for(int i = 0; i < mesh.vertices.size(); ++i) {
        mesh.initPos[i] = mesh.vertices[i].position;
    }


    return true;
}

uint MxMeshGmshImporter::addGmshVertex(const Gmsh::Node& node) {
    auto iter = vertexMap.find(node.id);

    if (iter != vertexMap.end()) {
        return iter->second;
    } else {
        uint id = mesh.addVertex(makeVertex(node.pos));
        vertexMap[node.id] = id;
        return id;
    }
}

/**
 * Create a Mechanica cell from a Gmsh Hexahedron, and add it to the
 * mesh. The Gmsh hexahedron vertices are ordered as such:
 *                          v
 *                   3----------2
 *                   |\     ^   |\
 *                   | \    |   | \
 *                   |  \   |   |  \
 *                   |   7------+---6
 *                   |   |  +-- |-- | -> u
 *                   0---+---\--1   |
 *                    \  |    \  \  |
 *                     \ |     \  \ |
 *                      \|      w  \|
 *                       4----------5
 * This means that we have to triangulate each face, and generate 12 partial
 * faces. Looking at the Gmsh hexahedron head-on, and unfolding, we get the
 * following arrangement. We order the partial faces with the following scheme.
 *                   3----------2
 *                   |         /|
 *                   |   0   /  |
 *                   |     /    |
 *                   |   /  1   |
 *                   | /        |
 *        3----------7----------6----------2----------3
 *        |         /|         /|         /|         /|
 *        |   2   /  |   4   /  |   6   /  |   8   /  |
 *        |     /    |     /    |     /    |     /    |
 *        |   /   3  |   /  5   |   /  7   |   /  9   |
 *        | /        | /        | /        | /        |
 *        0----------4----------5----------1----------0
 *                   |         /|
 *                   |  10   /  |
 *                   |     /    |
 *                   |   /  11  |
 *                   | /        |
 *                   0----------1
 * Important to pay attention to the triangle winding, we use CCW so each partial
 * face must be ordered accordingly. To get the normals to face correctly, they
 * need to point outwards. So, e.g. with pf[0], we have {7,2,3}, pf[1]={7,6,2},
 * pf[2]={7,3,0}... The start pos is not important as long as the winding is correct.
 */
void MxMeshGmshImporter::addCell(const Gmsh::Hexahedron& val) {
    //std::cout << "adding cell " << val.id << std::endl;

    // node indices mapping in the MxMesh vertices.
    uint vertexIds[8];
    MxCell &cell = mesh.createCell();

    //for (auto i : gmsh.nodes) {
    //    std::cout << "node id: " << i.first;
    //    std::cout << " {" << i.second.id << " {" << i.second.pos[0] << ", " << i.second.pos[1] << ", " << i.second.pos[2] << "}}" << std::endl;
    //}

    // grab the node positions out of the gmsh and add them to our mesh
    for(int i = 0; i < 8; ++i) {
        const Gmsh::Node &node = gmsh.nodes[val.nodes[i]];
        vertexIds[i] = addGmshVertex(node);
    }

    typedef VertexIndices VI;
    typedef PTriangleIndices PI;

    // make the boundary partial faces
    mesh.createPartialTriangle(nullptr, cell, VI{{vertexIds[7], vertexIds[2], vertexIds[3]}}, PI{{1,2,8}});

    mesh.createPartialTriangle(nullptr, cell, VI{{vertexIds[7], vertexIds[6], vertexIds[2]}}, PI{{0,4,6}});

    mesh.createPartialTriangle(nullptr, cell, VI{{vertexIds[0], vertexIds[7], vertexIds[3]}}, PI{{3,9,0}});

    mesh.createPartialTriangle(nullptr, cell, VI{{vertexIds[0], vertexIds[4], vertexIds[7]}}, PI{{2,4,10}});

    mesh.createPartialTriangle(nullptr, cell, VI{{vertexIds[4], vertexIds[6], vertexIds[7]}}, PI{{3,5,1}});

    mesh.createPartialTriangle(nullptr, cell, VI{{vertexIds[4], vertexIds[5], vertexIds[6]}}, PI{{4,6,10}});

    mesh.createPartialTriangle(nullptr, cell, VI{{vertexIds[5], vertexIds[2], vertexIds[6]}}, PI{{5,7,1}});

    mesh.createPartialTriangle(nullptr, cell, VI{{vertexIds[5], vertexIds[1], vertexIds[2]}}, PI{{6,8,11}});

    mesh.createPartialTriangle(nullptr, cell, VI{{vertexIds[1], vertexIds[3], vertexIds[2]}}, PI{{9,0,7}});

    mesh.createPartialTriangle(nullptr, cell, VI{{vertexIds[1], vertexIds[0], vertexIds[3]}}, PI{{8,11,2}});

    mesh.createPartialTriangle(nullptr, cell, VI{{vertexIds[0], vertexIds[5], vertexIds[4]}}, PI{{11,5,3}});

    mesh.createPartialTriangle(nullptr, cell, VI{{vertexIds[0], vertexIds[1], vertexIds[5]}}, PI{{7,10,9}});
}
