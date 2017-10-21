/*
 * MxMeshCore.h
 *
 *  Created on: Oct 3, 2017
 *      Author: andy
 */

#ifndef SRC_MXMESHCORE_H_
#define SRC_MXMESHCORE_H_

#include "mechanica_private.h"
#include <vector>
#include <array>
#include <algorithm>
#include <Magnum/Magnum.h>
#include <Magnum/Math/Vector3.h>


namespace Magnum {
/** @brief Three-component unsigned integer vector */
typedef Math::Vector3<UnsignedShort> Vector3us;
}

template<typename Container, typename Value>
bool contains(const Container& container, const Value& value) {
    return std::find(container.begin(), container.end(), value) != container.end();
}

template<typename Container, typename Value>
void remove(Container& cont, const Value& val) {
    auto it = std::find(cont.begin(), cont.end(), val);
    if(it != cont.end()) {
        cont.erase(it);
    }
}


using namespace Magnum;

struct MxVertex;
typedef MxVertex* VertexPtr;

typedef std::array<VertexPtr, 3> VertexIndices;

struct MxTriangle;
typedef MxTriangle *TrianglePtr;


struct MxPartialTriangle;
typedef MxPartialTriangle *PTrianglePtr;

typedef std::array<PTrianglePtr, 3> PartialTriangles;

struct MxCell;
typedef MxCell *CellPtr;

struct MxFacet;
typedef MxFacet *FacetPtr;

struct MxMesh;
typedef MxMesh *MeshPtr;


struct MxMeshNode {
    MxMeshNode(MeshPtr m) : mesh{m} {};

protected:
    MeshPtr mesh;
};

struct MxVertex {
    Magnum::Vector3 position;
    Magnum::Vector3 velocity;
    Magnum::Vector3 force;

    // one to many relationship of vertex -> triangles
    std::vector<TrianglePtr> triangles;

    // one to many relationship of vertex -> facets
    std::vector<FacetPtr> facets;
};

struct MxVertexAttribute {

    enum Id : ushort {Position, Normal};

    // the offset in the given vertex buffer memory block of where to write
    // the attribute
    ushort offset;

    // the id of this attribute. The id is simply an identifier that means something
    // to the MxMesh. This id does not have anything to do with the shader location id.
    // we keep separate ids because the MxMesh will often have many more attributes than
    // a renderer will want to display at any one time. Renderers typically display only
    // a subset of the available attributes. Attributes are things like scalar fields
    // attached to vertices, vertex position, normal, velocity, acceleration, etc...
    Id id;
};




#endif /* SRC_MXMESHCORE_H_ */
