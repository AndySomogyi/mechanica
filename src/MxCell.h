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
#include <Magnum/Magnum.h>
#include <Magnum/Math/Vector3.h>


namespace Magnum {
/** @brief Three-component unsigned integer vector */
typedef Math::Vector3<UnsignedShort> Vector3us;
}

using namespace Magnum;




struct MxMesh;

/**
 * Use unsigned as indexes, saves memory space, but need to define
 * what exactly an invalid index is
 */

template <class T>
inline bool is_valid(T val) {return val < std::numeric_limits<T>::max();};

template <class T>
inline T invalid() {return std::numeric_limits<T>::max();};

template <>
inline Vector3us invalid<Vector3us>() {return {invalid<ushort>(), invalid<ushort>(), invalid<ushort>()};};

struct MxMeshVertex {
    Vector3 position;
    Vector3 velocity;
    Vector3 acceleration;
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


/**
 * A partial face data structure, represents 1/2 of a triangular face. This represents the
 * side of the face that belongs to a cell struct. Partial faces share geometry (the vertex
 * indices at each apex), but have local attributes.
 *
 * The vertex index ordering is different in in each side of the partial face pair. We construct
 * the index ordering on each half such that the vertex winding causes the face normal to
 * point outwards from the cell. So, in each partial face half, the face normal points towards
 * the other cell.
 *
 * ** Face Dynamics **
 * There are a number of different options for calculating the time evolution of each vertex,
 * and by time evolution, we're strictly speaking about geometry here, not the kind of integration
 * we're using.
 *
 * We can choose to represent mass at either the vertex, or the face. As discussed elsewhere,
 * mass at the vertex is fundamentally flawed as most others implement it. The vertex simply
 * defines a point in space. The mass associated with that point is proportional to the
 * surrounding facets. We therefore choose to represent mass at the facet, rather than the
 * vertex. Consider arguments of the facet size changing. The mass is always conserved.
 *
 * Now, what exactly do we want to calculate the time evolution of. We can choose to either
 * accumulate force at the vertex, or the facet. A vertex based approach is troublesome because
 * we first need to calculate the mass, then, sum the force from the neighboring facets.
 *
 * We can represent the tendency for a facet to maintain it's area with a force acting
 * on it's neighbor's center of geometry towards the facet's own COG. This approach mimics
 * three harmonic bonded interactions in MD. The tendency for pairs of adjacent faces to maintain
 * a fixed angle is represented with a angle spring, this imparts a torque to the facet
 * normal vectors. Volume preservation / pressure can be represented with a force acting
 * at the facets COG, oriented toward's the facet's normal direction.
 *
 * In order to calculate the bending between facets, we need to know the facet's orientation,
 * we need to know it's normal vector. A triangle normal is relatively easy to calculate,
 * however the normal is a function of all three triangle positions, and it can be costly
 * to read them. Plus, the normal is required for all three side angle calculations. Thus,
 * we add a normal vector state variable to the partial face, even though this redundant,
 * it speeds up calculations.
 *
 * It is doubly redundant to attach a normal to each partial face. However, this approach is
 * simpler and quicker to implement, and in the future, we will adopt a more optimal strategy
 * of shared normals, where we can store the normals in a single array of the same size
 * as the number of faces. Each face can index this normal array via an integer index, but we
 * can also use the sign of the index as a normal sign multiplier. It the index is positive,
 * we multiply the normal by 1, if negative, by -1. In either case, the normal are indexed
 * by the abs value of the normal integer index.
 *
 * The centroid
 *
 */
struct MxPartialFace {

    /**
     * indices of the 3 vertices in the MxMesh that make up this partial face,
     * in the correct winding order
     */
    Vector3ui vertices;


    /**
     * index of the three neighbors of this face, these
     * neighbors are indices in the faces of the MxCell
     *
     * Don't expect ever to be more than 65,000 faces in a cell
     */
    Vector3us neighbors;

    /**
     * index of the neighboring cell that this partial face shares
     * a face with. We keep track of the neighboring cell here, this
     * lets the containing cell enumerate all neighboring cells via
     * connectivity through the partial faces.
     */
    uint neighborCell;

    /**
     * index in the neighboring cell's boundary array of the mirroring
     * partial face that matches this face.
     */
    ushort mirrorFace;

    double mass;


    /**
     * Last field in this struct, create a struct where the number of fields
     * are determined at runtime, and lets us allocate the entire struct in
     * a single contiguous memory block.
     *
     * But to get things working quickly, we'll just use a std::vector here
     * for now, lets us get the rest of the system up an running quickly
     * with just std::vector.
     */
    std::vector<double> fields;

    MxPartialFace(Vector3ui const& vert):
        vertices{vert},
        neighbors{invalid<ushort>(),invalid<ushort>(),invalid<ushort>()},
        neighborCell{invalid<uint>()},
        mirrorFace{invalid<ushort>()},
        mass{0}
    {
    }

    MxPartialFace():
          vertices{invalid<uint>(), invalid<uint>(), invalid<uint>()},
          neighbors{invalid<ushort>(),invalid<ushort>(),invalid<ushort>()},
          neighborCell{invalid<uint>()},
          mirrorFace{invalid<ushort>()},
          mass{0}
      {
      }
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
 */
struct MxCell {

    /**
     * the closed set of faces that define the boundary of this cell
     */
    std::vector<MxPartialFace> boundary;

    /**
     * fields that belong to this cell.
     */
    std::vector<double> fields;

    /**
     * iterate over the boundary partial faces and connect them all together.
     * This should be used when a new cell is created from a set of indices.
     *
     * Returns true if the boundary connected successfully, false if the
     * boundary in non-manifold.
     */
    bool connectBoundary(MxMesh& mesh);

    enum VolumeMethod { ConvexTrapezoidSum, GeneralDivergence };

    /**
     * Calculate the volume of this cell.
     */
    float volume(VolumeMethod vm = ConvexTrapezoidSum);

    /**
     * calculate the total area
     */
    float area();


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
    void vertexAtributeData(MxMesh& mesh, const std::vector<MxVertexAttribute> &attributes, uint vertexCount, uint stride, void* buffer);

    void indexData(uint indexCount, uint* buffer);

    void dump();

    void writePOV(std::ostream &out);
};

#endif /* SRC_MXCELL_H_ */
