/*
 * MxCell.h
 *
 *  Created on: Jul 7, 2017
 *      Author: andy
 */

#ifndef SRC_MXCELL_H_
#define SRC_MXCELL_H_

#include "mechanica_private.h"
#include <vector>
#include <array>
#include <Magnum/Magnum.h>
#include <Magnum/Math/Vector3.h>
#include "MxMeshCore.h"
#include "MxTriangle.h"
#include "MxFacet.h"
#include "MeshIterators.h"
#include "MeshRelationships.h"


struct MxCellRendererType : MxObject {

};


struct MxCellRenderer : MxObject {

    virtual HRESULT invalidate() = 0;
    virtual ~MxCellRenderer() {};
};


struct MxCellType : MxType {


    /**
     * Store the stoichiometry matrix in the type, initially Mechanica will
     * not support time-dependent stochiometries.
     */
};


/**
 * Represents a closed region of space. This corresponds spatially closely to the
 * concept of plant or animal cells, or rooms in a building. Each cell has a
 * well-defined boundary and a well-defined bounded spatial region. The boundary
 * of a cell is a fully-connected manifold surface.
 *
 * The MxCell maintains a set of local partial faces to represent the boundary surface
 * triangles. This cell knows what's inside and outside, hence it knows which way to order
 * the index winding so that the normal points the correct way.
 *
 * Nomenclature:
 *    *neighboring* partial faces are those that are in this cell's local surface
 *    manifold. There is a direct path from any partial face to any other partial face
 *    in each cell's surface.
 *
 *    An *adjacent* partial triangle is the located on the neighboring cell that is
 *    in direct contact with a partial face on this cell.
 *
 * A cell also references a set of state variables, typically chemical concentrations.
 * The state variables (state vector) block is owned by the top level mesh (later block
 * for multi-threaded), and the cell has a pointer to this memory. The cell's type object
 * has descriptors for the memory layout of the state vector. Derived types need to
 * calculate rate of change of the state vector.
 *
 * The way we do v-tables, derived types can contain objects, and stuff their
 * v-tables in the main v-table to do containment correctly.
 */
struct MxCell : MxObject, MxMeshNode {

    MxCell(MxCellType *type, MeshPtr msh, MxReal *stateVector) :
        MxObject{type}, MxMeshNode{msh}, stateVector{stateVector} {};

    /**
     * the closed set of faces that define the boundary of this cell
     */
    std::vector<struct MxPartialTriangle*> boundary;

    /**
     * A cell has one of more facets. Each facet defines a shared 2D region
     * of space between two different cells. If one cell is completely contained
     * within another cell, then it has one facet.
     */
    std::vector<FacetPtr> facets;

    /**
     * Pointer to the vector of state variables that belong to this cell. The state
     * vector memory is owned by the mesh, and this is an offset into the main
     * memory block.
     */
     MxReal *stateVector;

    /**
     * iterate over the boundary partial faces and connect them all together.
     * This should be used when a new cell is created from a set of indices.
     *
     * Returns true if the boundary connected successfully, false if the
     * boundary in non-manifold.
     */
    bool manifold() const;

    enum VolumeMethod { ConvexTrapezoidSum, GeneralDivergence };


    /**
     * Sometimes a mesh may have separate vertices for each triangle corner, even
     * though triangles may be neighboring and sharing a vertex. This occurs when
     * the triangle has it's own vertex attributes (values, normal, etc...), other times
     * a mesh may completely share a vertex between neighboring triangles and have
     * per-vertex normals ant attributes.
     */
    inline uint faceCount() { return boundary.size(); }

    /**
     * Even though we share position for vertices, each face has
     * different color, normal, etc... so need separate vertices
     * for each face. Not a problem because we compute these, don't
     * waste memory.
     */
    inline uint vertexCount() {return 3 * boundary.size(); };

    /**
     * Write vertex attributes to a supplied buffer. The given pointer is typically
     * returned from OpenGl, and points to a block of write-through memory, the pointer
     * is *write only*.
     *
     * @param stride: size in bytes of each element in the vertex buffer.
     */
    void vertexAtributeData(const std::vector<MxVertexAttribute> &attributes,
            uint vertexCount, uint stride, void* buffer);


    /**
     * Adds an orphaned triangle to this cell. The triangle must have at least one free
     * cell slot (it can only be attached to at most one existing cell).
     *
     * Searches the existing triangles, and examines if the new tri is adjacent
     * (shares a pair of vertices) with existing triangles, then this method connects
     * the partial faces of the existing and new triangle.
     *
     * If the new triangle is already attached to an existing cell, then this method
     * will either add the tri to the existing face (if one exists), or it will generate
     * a new face, and append that face to both this cell, and the other cell to which
     * the triangle incident to.
     *
     * @param index, specifies the the orientation of the triangle, must be either
     * 0 or 1. A 0 means the that the triangle winding orients the normal away from the
     * cell, and the cell goes in the MxTriangle::cell[0] position, a 1 means the
     * triangle winding is backwards, and must go in the MxTriangle::cells[1] slot. Must
     * go in the correct slot so the normal gets correctly calculated.
     */
    HRESULT appendChild(TrianglePtr tri, int index);


    /**
     * Removes the triangle from this cell, the triangle is then no-longer attached
     * to this cell. If the triangle is not attached to any other cell, it becomes
     * orphaned.
     *
     * Does not modify or alter geometry, removing a triangle from a cell will
     * likely leave a hole in the cell, making such that the cell is no longer
     * a manifold surface.
     *
     * Warning, volume calculations for non-manifold cells will not yield a
     * correct value, volume is undefined for non-manifold cells.
     */
    HRESULT removeChild(TrianglePtr tri);


    /**
     * Inform the cell that the vertex positions have changed. Causes the
     * cell to recalculate area and volume, also inform all contained objects.
     */
    HRESULT positionsChanged();

    void dump();

    void writePOV(std::ostream &out);

    /**
     * sum of all of the triangle areas.
     */
    float area = 0;

    /**
     * The volume of each cell is calculated using the divergence theorem, each triangle
     * contributes a bit of volume to each cell. We compute the volume contribution here
     * and then separately sum all of the volume contributions for each cell.
     *
     * The total volume of each cell is
     *
     * 1/3 \Sum \left( a_i/3 * \left( N \dot (r1 + r2 + r3\right)\right)
     *
     * where r1, r2, r3 are the position vectors for each vertex. Note that
     * 1/3 * (r1 + r2 + r3) is just the centroid of each triangle, which is already
     * computed by the triangle's positionChanged method, so to compute the volume contribution,
     * we just dot the centroid with the triangle's normal vector.
     */
    float volume = 0;

    Vector3 centroid = {0., 0., 0.};

    uint32_t id = 0;

    MxCellRenderer *renderer = nullptr;
};

#endif /* SRC_MXCELL_H_ */
