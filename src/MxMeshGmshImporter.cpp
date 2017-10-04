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

typedef VertexIndices VI;

static inline Magnum::Vector3 makeVertex(const double pos[]) {
    return Vector3{float(pos[0]), float(pos[1]), float(pos[2])};
}



bool MxMeshGmshImporter::read(const std::string& path) {

    gmsh = Gmsh::Mesh::read(path);

    for(auto& element : gmsh.elements) {
        switch(element.type) {
        case ElementType::Hexahedron: {
            addCell(element.get<Gmsh::Hexahedron>());
            break;
        }
        case ElementType::Prism: {
            addCell(element.get<Gmsh::Prism>());
            break;
        }
        default: continue;
        }
    }

    mesh.initPos.resize(mesh.vertices.size());

    for(int i = 0; i < mesh.vertices.size(); ++i) {
        mesh.initPos[i] = mesh.vertices[i]->position;
    }


    return true;
}

VertexPtr MxMeshGmshImporter::addGmshVertex(const Gmsh::Node& node) {
    auto iter = vertexMap.find(node.id);

    if (iter != vertexMap.end()) {
        VertexPtr p = iter->second;
        assert(mesh.valid(p));
        return p;
    } else {
        VertexPtr id = mesh.createVertex(makeVertex(node.pos));
        vertexMap[node.id] = id;
        assert(mesh.valid(id));
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
 * following arrangement. A problem with uniformly spiting each side into triangles
 * along the same axis (i.e. always from vert 7 to vert 8 as in below).
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
 * is that the side may be squashed, and we get low-quality triangles (long and
 * skinny), so search for the shortest diagonal, and split there. Sometimes
 * we may get the top face split like above, other times we may get:
 *                   3----------2
 *                   |\         |
 *                   |  \   0   |
 *                   |    \     |
 *                   |  1   \   |
 *                   |        \ |
 *                   7----------6
 *
 * Important to pay attention to the triangle winding, we use CCW so each partial
 * face must be ordered accordingly. To get the normals to face correctly, they
 * need to point outwards. So, e.g. with pf[0], we have {7,2,3}, pf[1]={7,6,2},
 * pf[2]={7,3,0}... The start pos is not important as long as the winding is correct.
 */
void MxMeshGmshImporter::addCell(const Gmsh::Hexahedron& val) {
    //std::cout << "adding cell " << val.id << std::endl;

    // node indices mapping in the MxMesh vertices.
    VertexPtr vertexIds[8];
    MxCell &cell = *mesh.createCell();

    //for (auto i : gmsh.nodes) {
    //    std::cout << "node id: " << i.first;
    //    std::cout << " {" << i.second.id << " {" << i.second.pos[0] << ", " << i.second.pos[1] << ", " << i.second.pos[2] << "}}" << std::endl;
    //}

    // grab the node positions out of the gmsh and add them to our mesh
    for(int i = 0; i < 8; ++i) {
        const Gmsh::Node &node = gmsh.nodes[val.nodes[i]];
        vertexIds[i] = addGmshVertex(node);
        assert(mesh.valid(vertexIds[i]));
    }

    // top face, vertices {2,3,7,6}
    addSquareFace(cell, {{vertexIds[2], vertexIds[3], vertexIds[7], vertexIds[6]}});

    addSquareFace(cell, {{vertexIds[7], vertexIds[3], vertexIds[0], vertexIds[4]}});

    addSquareFace(cell, {{vertexIds[6], vertexIds[7], vertexIds[4], vertexIds[5]}});

    addSquareFace(cell, {{vertexIds[2], vertexIds[6], vertexIds[5], vertexIds[1]}});

    addSquareFace(cell, {{vertexIds[3], vertexIds[2], vertexIds[1], vertexIds[0]}});

    addSquareFace(cell, {{vertexIds[5], vertexIds[4], vertexIds[0], vertexIds[1]}});

    assert(cell.manifold() && "Cell is not manifold");

    assert(mesh.valid(&cell));
}

void MxMeshGmshImporter::addSquareFace(MxCell& cell, const std::array<VertexPtr, 4>& verts) {
    
    assert(mesh.valid(verts[0]));
    assert(mesh.valid(verts[1]));
    assert(mesh.valid(verts[2]));
    assert(mesh.valid(verts[3]));

    float ne = (verts[0]->position - verts[2]->position).length();
    float nw = (verts[1]->position - verts[3]->position).length();
    if (nw > ne) {
        // nw is longer, split along ne axis
        //mesh.createPartialTriangle(nullptr, cell, VI{{verts[2], verts[1], verts[0]}});
    		createTriangleForCell(VI{{verts[0], verts[1], verts[2]}}, &cell);
    		createTriangleForCell(VI{{verts[2], verts[3], verts[0]}}, &cell);
    } else {
    		createTriangleForCell(VI{{verts[1], verts[2], verts[3]}}, &cell);
    		createTriangleForCell(VI{{verts[3], verts[0], verts[1]}}, &cell);
    }
}


/**
 * Add a 'Prism' element
 *
 *                 w
 *               ^
 *               |
 *               3
 *             ,/|`\
 *           ,/  |  `\
 *         ,/    |    `\
 *        4------+------5
 *        |      |      |
 *        |    ,/|`\    |
 *        |  ,/  |  `\  |
 *        |,/    |    `\|
 *       ,|      |      |\
 *     ,/ |      0      | `\
 *    u   |    ,/ `\    |    v
 *        |  ,/     `\  |
 *        |,/         `\|
 *        1-------------2
 *
 *
 *
 */
void MxMeshGmshImporter::addCell(const Gmsh::Prism& val) {
    // node indices mapping in the MxMesh vertices.
    VertexPtr vertexIds[6];
    CellPtr cell = mesh.createCell();

    //for (auto i : gmsh.nodes) {
    //    std::cout << "node id: " << i.first;
    //    std::cout << " {" << i.second.id << " {" << i.second.pos[0] << ", " << i.second.pos[1] << ", " << i.second.pos[2] << "}}" << std::endl;
    //}

    // grab the node positions out of the gmsh and add them to our mesh
    for(int i = 0; i < 6; ++i) {
        const Gmsh::Node &node = gmsh.nodes[val.nodes[i]];
        vertexIds[i] = addGmshVertex(node);
    }

    // create the two top and bottom faces.
    createTriangleForCell(VI{{vertexIds[0], vertexIds[2], vertexIds[1]}}, cell);
    createTriangleForCell(VI{{vertexIds[3], vertexIds[4], vertexIds[5]}}, cell);

    addSquareFace(*cell, {{vertexIds[5], vertexIds[4], vertexIds[1], vertexIds[2]}});
    addSquareFace(*cell, {{vertexIds[4], vertexIds[3], vertexIds[0], vertexIds[1]}});
    addSquareFace(*cell, {{vertexIds[3], vertexIds[5], vertexIds[2], vertexIds[0]}});
    
    assert(cell->manifold() && "Cell is not manifold");

    assert(mesh.valid(cell));
}

void MxMeshGmshImporter::createTriangleForCell(
		const std::array<VertexPtr, 3>& verts, CellPtr cell) {
	TrianglePtr tri = mesh.findTriangle(verts);
	if (!tri) {
		tri = mesh.createTriangle(nullptr, verts);
	}

	assert(tri);
	cell->appendChild(tri);
}
