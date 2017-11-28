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

MxCell *MxMeshGmshImporter::createCell(Gmsh::ElementType type, int id) {
    MxCellType *cellType = (elementCellTypeHandler) ? elementCellTypeHandler(type, id) : nullptr;
    return mesh.createCell(cellType);
}



HRESULT MxMeshGmshImporter::read(const std::string& path) {

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

    return mesh.positionsChanged();
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
    MxCell &cell = *createCell(Gmsh::ElementType::Hexahedron, val.id);
    cell.id = cellId++;

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
    addSquareFacet(cell, {{vertexIds[2], vertexIds[3], vertexIds[7], vertexIds[6]}});

    addSquareFacet(cell, {{vertexIds[7], vertexIds[3], vertexIds[0], vertexIds[4]}});

    addSquareFacet(cell, {{vertexIds[6], vertexIds[7], vertexIds[4], vertexIds[5]}});

    addSquareFacet(cell, {{vertexIds[2], vertexIds[6], vertexIds[5], vertexIds[1]}});

    addSquareFacet(cell, {{vertexIds[3], vertexIds[2], vertexIds[1], vertexIds[0]}});

    addSquareFacet(cell, {{vertexIds[5], vertexIds[4], vertexIds[0], vertexIds[1]}});

    assert(cell.manifold() && "Cell is not manifold");

    assert(cell.positionsChanged() == S_OK);
}

void MxMeshGmshImporter::addSquareFacet(MxCell& cell, const std::array<VertexPtr, 4>& verts) {

    assert(mesh.valid(verts[0]));
    assert(mesh.valid(verts[1]));
    assert(mesh.valid(verts[2]));
    assert(mesh.valid(verts[3]));

    FacetPtr facet = mesh.findFacet(verts);

    if(facet) {
        if(incident(facet, mesh.rootCell())) {
            mesh.rootCell()->removeChild(facet);
        }
        assert(facet->cells[0] == nullptr || facet->cells[1] == nullptr);
    } else {

        facet = mesh.createFacet(nullptr);

        float ne = (verts[0]->position - verts[2]->position).length();
        float nw = (verts[1]->position - verts[3]->position).length();
        if (nw > ne) {
            // nw is longer, split along ne axis
            // mesh.createPartialTriangle(nullptr, cell, VI{{verts[2], verts[1], verts[0]}});
            assert(Math::dot(
                   Math::normal(verts[0]->position, verts[1]->position, verts[2]->position),
                   Math::normal(verts[2]->position, verts[3]->position, verts[0]->position)) > 0);
                createTriangleForFacet(VI{{verts[0], verts[1], verts[2]}}, facet);
                createTriangleForFacet(VI{{verts[2], verts[3], verts[0]}}, facet);
        } else {
            assert(Math::dot(
                   Math::normal(verts[1]->position, verts[2]->position, verts[3]->position),
                   Math::normal(verts[3]->position, verts[0]->position, verts[1]->position)) > 0);
                createTriangleForFacet(VI{{verts[1], verts[2], verts[3]}}, facet);
                createTriangleForFacet(VI{{verts[3], verts[0], verts[1]}}, facet);
        }
    }

    cell.appendChild(facet);
    if(facet->cells[1] == nullptr) {
        mesh.rootCell()->appendChild(facet);
    }

    assert(facet->cells[0] && facet->cells[1]);
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
    CellPtr cell = createCell(Gmsh::ElementType::Prism, val.id);

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

    addSquareFacet(*cell, {{vertexIds[5], vertexIds[4], vertexIds[1], vertexIds[2]}});
    addSquareFacet(*cell, {{vertexIds[4], vertexIds[3], vertexIds[0], vertexIds[1]}});
    addSquareFacet(*cell, {{vertexIds[3], vertexIds[5], vertexIds[2], vertexIds[0]}});

    assert(cell->manifold() && "Cell is not manifold");

    assert(mesh.valid(cell));

    assert(cell->positionsChanged() == S_OK);
}

void MxMeshGmshImporter::createTriangleForCell(
        const std::array<VertexPtr, 3>& verts, CellPtr cell) {
    TrianglePtr tri = mesh.findTriangle(verts);
    if(tri) {
        if(incident(tri, mesh.rootCell())) {
            assert(mesh.rootCell()->removeChild(tri) == S_OK);
        }
    }
    else {
        tri = mesh.createTriangle(nullptr, verts);
    }

    assert(tri);
    assert(tri->cells[0] == nullptr || tri->cells[1] == nullptr);

    tri->mass = density * tri->area;

    Vector3 meshNorm = Math::normal(verts[0]->position, verts[1]->position, verts[2]->position);
    float orientation = Math::dot(meshNorm, tri->normal);
    cell->appendChild(tri, orientation > 0 ? 0 : 1);
}

void MxMeshGmshImporter::createTriangleForFacet(
        const std::array<VertexPtr, 3>& verts, FacetPtr facet) {
    TrianglePtr tri = mesh.findTriangle(verts);

    assert(tri == nullptr && "triangle already exists when creating facet");

    // creates a new triangle based on the winding order of the vertices.
    tri = mesh.createTriangle(nullptr, verts);

    assert(tri);
    assert(tri->cells[0] == nullptr || tri->cells[1] == nullptr);

    tri->mass = density * tri->area;

    facet->appendChild(tri);
}
