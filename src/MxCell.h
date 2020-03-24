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
#include "Magnum/Math/Color.h"
#include <Magnum/Math/Vector3.h>
#include <Magnum/Math/Matrix3.h>
#include <MxPolygon.h>
#include "MxMeshCore.h"
#include "MeshRelationships.h"


struct MxCellRendererType : CObject {

};


struct MxCellRenderer : CObject {

    virtual HRESULT invalidate() = 0;
    virtual ~MxCellRenderer() {};
};

CAPI_DATA(struct CType*) MxCell_Type;

struct MxCellType : CType {

    MxCellType(const char* name, CType *type)  {};

    //MxCellType() : CType{CType_Type} {};

    /**
     * Store the stoichiometry matrix in the type, initially Mechanica will
     * not support time-dependent stochiometries.
     */


    /**
     * TODO: hideously bad design, this is a complete and total fucking hack, fix
     * this shit at soon as possible.
     *
     * Color has no business here, cell rendering should be delegated completely to a
     * plugable cell renderer object. The cell renderer would likely be a closure of
     * some sort.
     */
    virtual Magnum::Color4 color(struct MxCell *cell) {
        return Magnum::Color4::yellow();
    }


    //Magnum::Color4 polygonEdgeColor = Magnum::Color4{{13.f/255, 161.f/255, 30.f/255, 1.f}};
    Magnum::Color4 polygonEdgeColor = Magnum::Color4{{98.f/255, 120.f/255, 168.f/255, 1.f}};
    Magnum::Color4 polygonCenterColor = Magnum::Color4{{73.f/255, 169.f/255, 163.f/255, 1.f}};
    Magnum::Color4 selectedEdgeColor = Magnum::Color4{{218.f/255, 142.f/255, 2.f/255, 1.f}};
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
struct MxCell : CObject, MxMeshNode {

    static CType *type() { return MxCell_Type; };

    MxCell(uint id, CType *type, MeshPtr msh, MxReal *stateVector, const std::string& nm = "")
    : CObject(MxCell_Type), MxMeshNode(msh, id)
    {

    };

    /**
     * the closed set of faces that define the boundary of this cell
     */
    std::vector<struct MxPartialPolygon*> surface;

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

    bool isValid() const;


    /**
     * is this the root cell?
     */
    bool isRoot() const;

    enum VolumeMethod { ConvexTrapezoidSum, GeneralDivergence };


    /**
     * Sometimes a mesh may have separate vertices for each triangle corner, even
     * though triangles may be neighboring and sharing a vertex. This occurs when
     * the triangle has it's own vertex attributes (values, normal, etc...), other times
     * a mesh may completely share a vertex between neighboring triangles and have
     * per-vertex normals ant attributes.
     */
    uint faceCount();

    /**
     * Even though we share position for vertices, each face has
     * different color, normal, etc... so need separate vertices
     * for each face. Not a problem because we compute these, don't
     * waste memory.
     */
    uint vertexCount();

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
     * Inform the cell that the topology changed (change in vertex or triangle
     * number or connectivity).
     */
    HRESULT topologyChanged();

    /**
     * Inform the cell that the vertex positions have changed. Causes the
     * cell to recalculate area and volume, also inform all contained objects.
     *
     * Assumes that the polygons have already had their positionsChanged function
     * called and their attributes are updated.
     */
    HRESULT positionsChanged();

    void dump() const;

    void writePOV(std::ostream &out);

    Vector3 centerOfMass() const;

    Vector3 radiusMeanVarianceStdDev() const;

    Matrix3 momentOfInertia() const;

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

    std::string name;

    Vector3 centroid = {0., 0., 0.};

    MxCellRenderer *renderer = nullptr;

    bool render = true;
    
private:


};

std::ostream& operator << (std::ostream& stream, const MxCell *cell);


/**
 * TODO, horribly bad design, need to completely reevaluate how we handle cell
 * rendering / vertex attributes.
 */
struct VertexAttribute
{
    Magnum::Vector3 position;
    Magnum::Vector3 normal;
    Magnum::Color4 color;
};

#endif /* SRC_MXCELL_H_ */
