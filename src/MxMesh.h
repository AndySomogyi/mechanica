/*
 * MxMesh.h
 *
 *  Created on: Jan 31, 2017
 *      Author: andy
 */

#ifndef _INCLUDE_MXMESH_H_
#define _INCLUDE_MXMESH_H_

#include <vector>
#include <list>
#include <deque>
#include <queue>

#include <Magnum/Magnum.h>
#include <Magnum/Mesh.h>
#include "mechanica_private.h"
#include "mdcore_single.h"

#include "MxCell.h"
#include "MxEdge.h"



/**
 * The type object for a universe cell.
 */
MxAPI_DATA(MxCellType) *MxUniverseCell_Type;

/**
 * The type object for a universe partial triangle.
 */
MxAPI_DATA(MxPartialTriangleType) *MxUniversePartialTriangle_Type;

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
 *
 * The mesh contains a hierarchy of elements. The most basic element is the vertex, this
 * is a single point that defines a position, velocity, and accumulates force. Three vertices
 * combine to form a triangle. Cells define a finite region of space that is bounded by
 * triangles.
 *
 * The mesh defines a special 'universe' cell, this is the first cell that is always created,
 * has an id of 0. The mesh also defines a special partial triangle, again with index 0 that
 * connects all of the exposed faces of non-universe cells to the universe. Every time a new
 * triangle or cell is created, it is automatically connected to the universe cell through the
 * special universe partial triangle. The universe partial triangle is connected to itself
 * through it's neighbors.
 */
struct MxMesh  {


    MxMesh();

    ~MxMesh();


    std::vector<Magnum::Vector3> initPos;


    /**
     * @brief Initialize an #engine with the given data.
     *
     * @param e The #engine to initialize.
     * @param origin An array of three doubles containing the Cartesian origin
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

    //MxMesh(const Vector3 &origin , const Vector3 &dim , const Vector3 &L ,
    //        double cutoff , unsigned int period , int max_type , unsigned int flags );


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
    void vertexAtributes(const std::vector<MxVertexAttribute> &attributes, uint vertexCount,
    		uint stride, void* buffer);

    int findVertex(const Magnum::Vector3 &pos, double tolerance = 0.00001);


    /**
     * Searches for a triangle which contains the given three vertices.
     */
    TrianglePtr findTriangle(const std::array<VertexPtr, 3> &vertexInd);


    /**
     * Searches for a facet joining cells a and b, returns if found. null otherwise.
     */
    FacetPtr findFacet(CellPtr a, CellPtr b);

    /**
     * Creates a new facet that connects cells a and b.
     */
    FacetPtr createFacet(MxFacetType *type, CellPtr a, CellPtr b);


    /**
     * Creates a new triangle for the given three vertices.
     *
     * returns a new, orphaned triangle.
     */
    TrianglePtr createTriangle(MxTriangleType *type, const std::array<VertexPtr, 3> &verts);

    /**
     * Creates a new empty cell and inserts it into the cell inventory.
     */
    CellPtr createCell(MxCellType *type = nullptr);

    void dump(uint what);

    void jiggle();

    std::tuple<Magnum::Vector3, Magnum::Vector3> extents();



    /**
     * inform the mesh that the vertex position was changed. This causes the mesh
     * to check if any adjoining edge lengths exceed the distance cutoffs.
     *
     * The mesh will then place them in a set of priority queues (based on distance), and
     * will process all of the offending edges.
     */
    HRESULT positionsChanged();

    /**
     * process all of the edges that violate the min/max cutoff distance constraints.
     */
    HRESULT processOffendingEdges();


    VertexPtr createVertex(const Magnum::Vector3 &pos);


    HRESULT collapseEdge(const MxEdge& edge);

    HRESULT splitEdge(const MxEdge &e);

    HRESULT collapseHTriangle(TrianglePtr tri);

    //HRESULT collapseHTriangleOld(MxTriangle &tri);

    HRESULT collapseIEdge(const MxEdge &edge);

    HRESULT collapseManifoldEdge(const MxEdge &e);

    bool valid(TrianglePtr p);

    bool valid(CellPtr c);

    bool valid(VertexPtr v);

    bool valid(PTrianglePtr p);

    CellPtr rootCell() {return _rootCell;};

    std::vector<TrianglePtr> triangles;
    std::vector<VertexPtr> vertices;
    std::vector<CellPtr> cells;
    std::deque<FacetPtr> facets;


private:




    /**
     * Splits a vertex located at the boundary of a facet. The given vertex gets removed from the
     * facet, and replaced with the vertices v0 and v1. Vertices v0 and v1 should be complete,
     * in that they should have their positions set. This method creates a new triangle,
     * and returns in it result.
     *
     * The newly formed edge is in the result triangle 0 and 1 indices. The caller is responsible
     * for splitting the all of the remaining faced incident to v. The method generates two new
     * partial triangles also, but the 0'th neighboring partial triangle index is left to nullptr,
     * the caller is responsible for attaching the new partial triangles.
     *
     * This method does not delete v, it only detaches v from the facet, and attaches v0 and
     * v1 to the facet.
     */
    TrianglePtr splitFacetBoundaryVertex(FacetPtr face, VertexPtr v, VertexPtr v0, VertexPtr v1);

    /**
     * Collapses a facet that contains a single triangle down to vertex. The facet,
     * and triangle are removed from the cell. First a new vertex is generated at the
     * center of the target triangle, and the neighboring facets, partial
     * triangles and triangles all get re-attached to this vertex. The given facet, and
     * it's contained triangle, partial triangle and vertices get orphaned, and the
     * caller is responsible for deleting them. The newly created vertex is returned in
     * result.
     */
    VertexPtr collapseCellSingularFacet(CellPtr cell, FacetPtr facet);

    /**
     * iterate over all of the triangles attached to vertex o, and
     * replace the triangle's o vertex with n.
     */
    void vertexReconnect(VertexPtr o, VertexPtr n);

    /**
     * Removes the triangle from the triangle list of the old vertex o,
     * and attaches it to the list of n. Updates tri to replace o with n.
     */
    void triangleVertexReconnect(MxTriangle &tri, VertexPtr o, VertexPtr n);

    /**
     * The triangle tri is going to be removed, with an edge collapse. Take the partial faces on
     * both sides and connect them to each other. Verifies that the tri is a manifold triangle.
     *
     * This only reconnect the partial faces. reconnectVertex must be used separately
     * to reconnect the triangles attached to a removed vertex. Removes the triangle
     * from the cell's list of partial faces.
     */
    void triangleManifoldDisconnect(const MxTriangle &tri, const MxEdge &edge);

    bool splitWedgeVertex(VertexPtr v0, VertexPtr nv0, VertexPtr nv1, MxCell* c0,
            MxCell* c1, MxTriangle *tri);


    // hack to check if the queue contains an element
    // the std::priority_queue container is protected, and we need to check
    // if the item is contained before inserting.
    template<class Compare>
    class EdgeQueue : public std::priority_queue<MxEdge, std::deque<MxEdge>, Compare> {
        typedef std::priority_queue<MxEdge, std::deque<MxEdge>, Compare>  _base;
    public:
        bool contains(const std::array<VertexPtr, 2> &value) const {
            for(auto i = _base::c.begin(); i < _base::c.end(); i++) {
                const MxEdge& e = *i;
                if(e == value) {
                    return true;
                }
            }
            return false;
        }

        void push(const std::array<VertexPtr, 2> &value) {
            if (!contains(value)) {
                _base::push(MxEdge{value[0], value[1]});
            }
        }
    };

    EdgeQueue<std::less<MxEdge>> shortEdges;
    EdgeQueue<std::greater<MxEdge>> longEdges;

    float shortCutoff = 0.2;
    float longCutoff = 0.5;

    CellPtr _rootCell;

    friend struct MxVertex;
    friend struct MxTriangle;
    friend struct MxFacet;
    friend struct MxEdge;
    friend struct MxCell;
};


#endif /* _INCLUDE_MXMESH_H_ */
