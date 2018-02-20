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
#include <set>
#include <Magnum/Magnum.h>
#include <Magnum/Math/Vector3.h>
#include <iostream>


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


using Vector3 = Magnum::Vector3;
using Color4 = Magnum::Color4;


using namespace Magnum;

struct MxVertex;
typedef MxVertex* VertexPtr;
typedef const MxVertex *CVertexPtr;

typedef std::array<VertexPtr, 3> VertexIndices;

struct MxTriangle;
typedef MxTriangle *TrianglePtr;
typedef const MxTriangle *CTrianglePtr;


struct MxPartialTriangle;
typedef MxPartialTriangle *PTrianglePtr;
typedef const MxPartialTriangle *CPTrianglePtr;

typedef std::array<PTrianglePtr, 3> PartialTriangles;

struct MxCell;
typedef MxCell *CellPtr;
typedef const MxCell *CCellPtr;

struct MxMesh;
typedef MxMesh *MeshPtr;
typedef const MxMesh *CMeshPtr;

// triangle container in the main mesh
typedef std::vector<TrianglePtr> TriangleContainer;

// triangle container for the vertices
typedef std::vector<TrianglePtr> Triangles;


struct MxMeshNode {
    MxMeshNode(MeshPtr m, uint _id) : id{_id}, mesh{m} {};

    const uint id;

protected:
    MeshPtr mesh;

    friend class MxVertex;
};

struct MxVertex {
    /**
     * The Mechanica vertex does not represent a point mass as in a traditional
     * particle based approach. Rather, the vertex here represents a region of space,
     * hence, we need to calculate the mass as a weighted sum of all the neighboring
     * triangles.
     *
     * TODO: We presently compute the mass in the MxTriangle::positionsChanged method.
     * This approach is not very cache friendly, will come up with a more optimal
     * solution in a later release.
     */
    float mass = 0;

    float area = 0;

    static float minForceDivergence;
    static float maxForceDivergence;

    MxVertex(float mass, float area, const Magnum::Vector3 &pos) :
        mass{mass}, area{area}, position{pos} {};

    Magnum::Vector3 position;
    Magnum::Vector3 velocity;

    float attr = 0;

     // one to many relationship of vertex -> triangles
    const Triangles &triangles() const {return _triangles;}

    const std::vector<CellPtr> &cells() const { return _cells; }

    std::set<VertexPtr> link() const;

    HRESULT removeTriangle(const TrianglePtr);

    HRESULT appendTriangle(TrianglePtr);

    /**
     * Inform the vertex that the cell of an attached triangle was changed.
     */
    HRESULT triangleCellChanged(TrianglePtr tri);

    /**
     * Find the first triangle that is incident to the given cell.
     * If no triangle is incident, returns null.
     */
    TrianglePtr triangleForCell(CCellPtr cell) const;

    uint id{0};

    /**
     * Area weighted vector that's the some area weighted sum of all
     * incident triangles for the given cell.
     */
    Magnum::Vector3 areaWeightedNormal(CCellPtr cell) const;



private:
     // one to many relationship of vertex -> triangles
    Triangles _triangles;

    // one to many relationship of vertex -> cells
    std::vector<CellPtr> _cells;

    void rebuildCells();

    /**
     * Get the mesh pointer.
     */
    MeshPtr mesh();

    /**
     * temporary hack for volume constraint.
     */
    Vector3 awc;
    friend class MxCell;
};

std::ostream& operator<<(std::ostream& os, CVertexPtr v);



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


/**
 * TODO: clean up the naming convention.
 *
 * An Edge is a lightweight pair of vertices, but an MxEdge is a heavy
 * struct that enumerates all the the triangle relationships.
 */
typedef std::array<VertexPtr, 2> Edge;

inline float length(const Edge& edge) {
    return (edge[0]->position - edge[1]->position).length();
}



#endif /* SRC_MXMESHCORE_H_ */
