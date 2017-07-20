/*
 * MxMesh.h
 *
 *  Created on: Jan 31, 2017
 *      Author: andy
 */

#ifndef _INCLUDE_MXMESH_H_
#define _INCLUDE_MXMESH_H_

#include <vector>
#include <Magnum/Magnum.h>
#include <Magnum/Mesh.h>
#include "mechanica_private.h"
#include "mdcore_single.h"

#include "MxCell.h"



/**
 * Internal implementation of MxObject
 *
 * This mesh structure stores all position, velocity and acceleration information
 * in a set of mdcore particles. The MxMesh
 *
 * Logically, the MxMesh exists to provide represent a collection of geometric information
 * (positions, connectivity, vertex attributes such as concentrations, etc... ), and
 * should not try to mix in time evolution here. However, we are using the mdcore library
 * as our particle dynamics engine, and it combines everything into a single top-level
 * struct called `engine`. In order to cleanly separate concerns, we need to refactor
 * this functionality apart at a later date. However, to get things working quickly,
 * I think we can just lump it all together here for now. So, for the time being,
 * we'll just keep a pointer to an mdcore engine and space here, and clean it up later.
 *
 *
 * Particles can have many different kinds of bonded relationships. Even in MD alone, we have
 * harmonic bonds, angles, dihedrals, etc... Other kinds of bonded relationships are transient
 * in time (non-bonded interactions). We need a way to efficiently render these.
 *
 * We can create a separate Magnum::Mesh for each one of these kinds of bonded relationships.
 * This is probably the most flexible approach as it lets us define new kinds of bonded
 * relationships in the future.
 *
 * We want to try to keep the mesh data (this class) as separate as possible from the rendering,
 * MxMeshRenderer. However, we also want to maximize performance. That means that we need
 * an efficient way of representing and copying mesh data between the mesh and the renderer.
 * The mesh renderer will keep track of all graphics card objects (vertex buffers, etc...),
 * and will query this object to check if anything has changed. This class also provides
 * info as to how frequently the various geometric objects change (TODO). But for now,
 * we'll assume everything can change frequently. This object currently will store and
 * manipulate data CPU side. In the future, we'd like to move it as much as possible
 * to OpenCL.
 *
 * Copying data to the graphics processor is tricky here because many things like particle
 * position info is not stored in contiguous arrays (mdcore stores it in chunks). This class
 * also will likely change internal data representations in the future. Some options are either
 * mapping / unmapping the graphics processor buffer memory (glMapBuffer/glUnmapBuffer), or
 * explicitly calling glBufferSubData. The map version could simply be implemented by passing
 * the pointer returned from the map operation to this object, and letting this object manage
 * all the vertex writing. The glBufferSubData approach would also work if this object
 * returned a vector of structs that contained vertex ranges. Here, each struct would have
 * a pointer to the memory block where the vertex is stored, and size of how many vertices are
 * in that block. This approach potentially could offer increased performance, but is more
 * complex to implement. Hence, in the initial version, we will use the map buffer approach,
 * where the mesh renderer gives this object a pointer of where to write the data to, and this
 * object takes care of all the copying from mdcore memory into the mapped graphics memory.
 *
 * Future versions could optimize performance further potentially by keeping a the graphics
 * mapped pointer during the start of the vertex position update cycle, and writing the
 * updated positions during the course of updates. This would only incur a single read for
 * the particle data, and likely would keep two cache lines for the writes (particle, and
 * graphics memory). Synchronization might be an issue though.
 *
 * We make extensive use of indices here instead of pointers because each pointer takes up
 * 64 bits, and we're going to store a lot of data here. Want to minimize memory usage, and
 * consequently, access time for memory. An index is just an offset in an array, which should
 * generate essentially the same code as a pointer dereference, just adds an offset to it.
 * CPU instructions are SIGNIFICANTLY faster that the memory access time.
 */
struct MxMesh  {

    MxMesh() {};

    /**
     * The set of triangle faces that define boundary of this cell. Each
     * one of these in an index into the vector of positions owned by the
     * MxMesh.
     */
    std::vector<Magnum::Vector3ui> faces;


    std::vector<MxMeshVertex> vertices;

    std::vector<Magnum::Vector3> initPos;


    std::vector<MxCell> cells;


    // mdcore engine (where geometry lives)
    engine mdEngine;


    Vector3 vertex(uint);

    /**
     * @brief Initialize an #engine with the given data.
     *
     * @param e The #engine to initialize.
     * @param origin An array of three doubles containing the cartesian origin
     *      of the space.
     * @param dim An array of three doubles containing the size of the space.
     * @param L The minimum cell edge length in each dimension.
     * @param cutoff The maximum interaction cutoff to use.
     * @param period A bitmask describing the periodicity of the domain
     *      (see #space_periodic_full).
     * @param max_type The maximum number of particle types that will be used
     *      by this engine.
     * @param flags Bit-mask containing the flags for this engine.
     *
     * @return #engine_err_ok or < 0 on error (see #engine_err).
     */

    MxMesh(const Vector3 &origin , const Vector3 &dim , const Vector3 &L ,
            double cutoff , unsigned int period , int max_type , unsigned int flags );

    /**
     * Not good design, reading from data files belongs in a separate class,
     * TODO refactor this later.
     *
     * Read the particles and connectivity info from a pair of psf and pdb files.
     */
    bool readPSF(const std::string& psfPath, const std::string& pdbPath);




    /**
     * Write vertex attributes into a given buffer.
     *
     * Challenges here are how do we return vertex attributes when only rendering a
     * subset of vertices, like say for a individual cell?
     *
     * Other issue, is if we keep attributes per vertex, this approach works well for free
     * particles, i.e. things like fluids, or *separate* cells. But what about when vertices are
     * shared between adjacent cells. One of the issues here is that each cell maintains
     * a set of attributes local to that cell, like say scalar fields that are specific to
     * certain cells. Even when cells are on contact, each cell typically maintains it's own
     * surface, with it's own attributes bound to it's surface.
     *
     * So, what we can do, is we can can have a set of *global* attributes for each vertex,
     * things like position, velocity, etc.., but each cell need to be able to assign it's
     * own attributes to vertices. Another question, is do we want per vertex, or per
     * face attributes?
     *
     * One idea is to think in terms of the dual of the vertex graph, the face graph. So,
     * instead of vertex dynamics, we have the concept of face dynamics. Graphics hardware
     * is really set up to only have per vertex attributes. One approach is instead of
     * sending all the vertices, instead send the face centroids and attributes, and have
     * a geometry shader generate a three vertices per centroid point. This approach
     * would cut down on the amount of data sent to the graphics processor, but at the
     * expense of increased computational requirements on the CPU. This is probably not that
     * bad, as the biggest bottleneck is not compute time, but rather memory access speed.
     *
     * Dynamics of the free particles is very different from meshed particles. Main differences
     * here is that there is no verlet list or spatial cell (in MD, we partition the world into
     * a regular grid of 'cells', and each one contains a set of particles. This speeds up
     * long-range force calculations, and enables efficient verlet list construction). But none
     * of this really has any need for meshed particles, as we know exactly what they're connected
     * to. We will later need a way to couple the meshed particles to the free particles. When we
     * connect them, we need a way to partition the free particles.
     *
     * Does it even make sense to use the mdcore engine for bound particles? Probably not, as the
     * dynamics are so different. What we can do, is add a new type of potential to the mdcore
     * engine to represent our surfaces. Maybe have an instance of the mdcore engine per MxCell
     * to represent free particles inside each cell?
     *
     * Anyway, our immediate goal is the implement vertex dynamics as quickly as possible.
     *
     * Observation is that we don't always have to render the shared surfaces, most of the
     * time, we only need to render the outside of the system, i.e. the medium facing
     * surfaces.
     *
     * Physically, I think it makes the most sense to represent amount of material per
     * face, rather than vertex. I think it makes the most sense to also have the
     * concept of mass at the face, rather than the vertex.
     *
     * Transport equations are simpler to solve with a face based approach, as each face
     * has exactly three neighboring faces, but each vertex may have anywhere from three
     * (in the case of the simplest possible manifold surface, the tetrahedron) to N
     * neighboring faces.
     *
     * The simplest way to implement mass-at-face dynamics, but we still still have the
     * concept of shared geometry is to have a single large array of vertices in the
     * top-level mesh, and continue with the original idea of the half-face data structure
     * which indexes this array. Now, should we have the vertices, or the faces move?
     * I think it's easier to move the vertices, i.e. solve the equations of motion
     * at the vertex rather than the face level. It's certainly possible to solve equations
     * of motion at the face level with an equation of the motion of the centroid of the
     * face.
     *
     * Each face experiences a net force on it, force from the inside and outside. As we
     * treat each face as a uniform solid, and internal pressure is uniform, force acts at
     * the centroid of the triangle, in the normal direction.
     *
     * We can partition each triangle into three equally sized subsections by splitting it
     * on it's centroid. Then, the total mass at each vertex is the 1/3 the sum of all
     * neighboring triangles. The total force at each vertex is also 1/3 the vector sum of
     * all the neighboring faces force vectors.
     *
     */
    void vertexAtributes(const std::vector<MxVertexAttribute> &attributes, uint vertexCount, uint stride, void* buffer);

    int findVertex(const Magnum::Vector3 &pos, double tolerance = 0.00001);

    /**
     * Adds a new vertex to this mesh, just pushes it to the end of the
     * list of verticies.
     */
    uint addVertex(const Magnum::Vector3 &pos);




    static const Magnum::Mesh::IndexType IndexType = Magnum::Mesh::IndexType::UnsignedInt;





    MxCell &createCell();


    void dump(uint what);

    void jiggle();

    std::tuple<Magnum::Vector3, Magnum::Vector3> extents();








};

#endif /* _INCLUDE_MXMESH_H_ */
