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



MxMesh MxMeshGmshImporter::read(const std::string& path) {

    mesh = MxMesh{};
    gmsh = Gmsh::Mesh::read(path);

    for(auto& element : gmsh.elements) {
        switch(element.type) {
        case ElementType::Hexahedron: addCell(element.get<Gmsh::Hexahedron>()); break;
        default: continue;
        }
    }

    return mesh;
}

uint MxMeshGmshImporter::addNode(const Gmsh::Node& node) {
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
    uint nodes[8];
    MxCell &cell = mesh.createCell();

    cell.boundary.resize(12);
    
    //for (auto i : gmsh.nodes) {
    //    std::cout << "node id: " << i.first;
    //    std::cout << " {" << i.second.id << " {" << i.second.pos[0] << ", " << i.second.pos[1] << ", " << i.second.pos[2] << "}}" << std::endl;
    //}

    // grab the node positions out of the gmsh and add them to our mesh
    for(int i = 0; i < 8; ++i) {
        const Gmsh::Node &node = gmsh.nodes[val.nodes[i]];
        nodes[i] = addNode(node);
    }

    // make the boundary partial faces
    cell.boundary[0].vertices = Vector3ui{nodes[7], nodes[2], nodes[3]};
    cell.boundary[0].neighbors = Vector3us{1,2,8};

    cell.boundary[1].vertices = Vector3ui{nodes[7], nodes[6], nodes[2]};
    cell.boundary[1].neighbors = Vector3us{0,4,6};

    cell.boundary[2].vertices = Vector3ui{nodes[0], nodes[7], nodes[3]};
    cell.boundary[2].neighbors = Vector3us{3,9,0};

    cell.boundary[3].vertices = Vector3ui{nodes[0], nodes[4], nodes[7]};
    cell.boundary[3].neighbors = Vector3us{2,4,10};

    cell.boundary[4].vertices = Vector3ui{nodes[4], nodes[6], nodes[7]};
    cell.boundary[4].neighbors = Vector3us{3,5,1};

    cell.boundary[5].vertices = Vector3ui{nodes[4], nodes[5], nodes[6]};
    cell.boundary[5].neighbors = Vector3us{4,6,10};

    cell.boundary[6].vertices = Vector3ui{nodes[5], nodes[2], nodes[6]};
    cell.boundary[6].neighbors = Vector3us{5,7,1};

    cell.boundary[7].vertices = Vector3ui{nodes[5], nodes[1], nodes[2]};
    cell.boundary[7].neighbors = Vector3us{6,8,11};

    cell.boundary[8].vertices = Vector3ui{nodes[1], nodes[3], nodes[2]};
    cell.boundary[8].neighbors = Vector3us{9,0,7};

    cell.boundary[9].vertices = Vector3ui{nodes[1], nodes[0], nodes[3]};
    cell.boundary[9].neighbors = Vector3us{8,11,2};

    cell.boundary[10].vertices = Vector3ui{nodes[0], nodes[5], nodes[4]};
    cell.boundary[10].neighbors = Vector3us{11,5,3};

    cell.boundary[11].vertices = Vector3ui{nodes[0], nodes[1], nodes[5]};
    cell.boundary[11].neighbors = Vector3us{7,10,9};
}
