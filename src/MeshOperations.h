/*
 * MeshOperations.h
 *
 *  Created on: Nov 20, 2017
 *      Author: andy
 */

#ifndef SRC_MESHOPERATIONS_H_
#define SRC_MESHOPERATIONS_H_

#include "MxMeshCore.h"
#include "MxEdge.h"

#include <random>

/**
 * An operation modifies a mesh, typically topologically.
 *
 * Operations get constructed with their arguments, then they
 * get re-ordered in the queue based on their energy cost. The values
 * of their arguments change as other operations modify the mesh.
 * Operations must be removed from the queue if their arguments
 * get removed from the queue.
 *
 * Basic responsibilities:
 *
 * * factory operation, given a triangle or edge, create a new operation.
 */



struct MeshOperation {

    MeshOperation(MeshPtr _mesh) : mesh{_mesh} {};

    /**
     * Apply this operation
     */
    virtual HRESULT apply() = 0;

    /**
     * lower, more negative energy operations are queued at a higher priority.
     */
    virtual float energy() const = 0;

    /**
     * does this operation depend on this triangle?
     */
    virtual bool depends(const TrianglePtr) const = 0;

    /**
     * does this operation depend on this vertex?
     */
    virtual bool depends(const VertexPtr) const = 0;

    virtual bool equals(const Edge& e) const = 0;


    bool  operator<(const MeshOperation& other) const {
        return energy() < other.energy();
    }

    virtual ~MeshOperation() {};

protected:
    MeshPtr mesh;
};


    /**
     * A manifold edge is an edge on a closed surface, it
     * resided between exactly two triangles. If the given edge
     * does not have one or two indicent triangles, returns a failure.
     *
     * To split an edge, it doesn't really matter if this edge is a
     * manifold edge or not. This method creates a new vertex at the
     * center of this edge, and splits each incident triangle into
     * two, and reconnects them. Each new triangle maintains the same
     * cell connectivity relationships.
     *
     * For every triangle connected to an edge like:
     *
     *
     *     edge.a
     *       *
     *       | \
     *       |   \
     *       |     \    B
     *       |       \
     *       |         \
     *       |          * c
     *    A  |         /
     *       |        /
     *       |       /
     *       |      /  C
     *       |     /
     *       |    /
     *       |   /
     *       |  /
     *       | /
     *       *
     *    edge.b
     *
     *       *
     *       | \
     *       |   \
     *       |     \    <- new triangle
     *       |       \
     *       |         \
     *       |      _ - * c
     *       |  _ -    /
     *    n  *-       /
     *       |       /
     *       |      /
     *       |     /    <- old triangle
     *       |    /
     *       |   /
     *       |  /
     *       | /
     *       *
     *
     *  * maintain a list of the newly created triangles.
     *  * for each existing triangle
     *        disconnect the old triangle from the top vertex
     *        disconnect the old triangle from the a-c edge
     *        create a new triangle
     *        attach the new tri to the old triangle, and the a-c edge
     *        add new tri to list
     *        add new tri to each cell that the old tri belongs to.
     *
     *  * for each new tri
     *        connect the tri to the next and prev tri around the n-a edge
     *
     */


struct RadialEdgeSplit : MeshOperation {
    RadialEdgeSplit(MeshPtr mesh, float longCutoff, const Edge& edge);

    static bool applicable(const Edge& e);

    /**
     * Apply this operation
     */
    virtual HRESULT apply();

    /**
     * lower, more negative energy operations are queued at a higher priority.
     */
    virtual float energy() const;

    /**
     * does this operation depend on this triangle?
     */
    virtual bool depends(const TrianglePtr) const;

    /**
     * does this operation depend on this vertex?
     */
    virtual bool depends(const VertexPtr) const;

    virtual bool equals(const Edge& e) const;

private:

    std::default_random_engine randEngine;
    std::uniform_real_distribution<float> uniformDist;

    float longCutoff;
    Edge edge;
};

struct RadialEdgeCollapse : MeshOperation {

    RadialEdgeCollapse(MeshPtr, float shortCutoff, const Edge&);

    static bool applicable(const Edge& e);

    /**
     * Apply this operation
     */
    virtual HRESULT apply();

    /**
     * lower, more negative energy operations are queued at a higher priority.
     */
    virtual float energy() const;

    /**
     * does this operation depend on this triangle?
     */
    virtual bool depends(const TrianglePtr) const;

    /**
     * does this operation depend on this vertex?
     */
    virtual bool depends(const VertexPtr) const;

    virtual bool equals(const Edge& e) const;

private:
    float shortCutoff;
    Edge edge;
};

struct EdgeFlip : MeshOperation {
    EdgeFlip(MeshPtr mesh, const Edge& endge);

    static bool applicable(const Edge& e);
};

/**
 * A priority queue of mesh operations.
 */
struct MeshOperations {



    MeshOperations(MeshPtr mesh, float shortEdgeCutoff, float longEdgeCutoff);

    /**
     * Inform the MeshOperations that a mesh object (likely positions) was changed,
     * Check if the mesh object triggers any operations and enqueue any triggered
     * operations. A mesh object can trigger an operation if an edge is too short
     * or long.
     */
    HRESULT positionsChanged(TriangleContainer::const_iterator begin,
            TriangleContainer::const_iterator end);

    /**
     * Inform the MeshOperations that a mesh object (likely positions) was changed,
     * Check if the mesh object triggers any operations and enqueue any triggered
     * operations. A mesh object can trigger an operation if an edge is too short
     * or long.
     */
    HRESULT valenceChanged(const VertexPtr vert);

    /**
     * a mesh object was deleted, remove any enqueued operations
     * that refer to this obj.
     */
    HRESULT removeDependentOperations(const TrianglePtr tri);


    /**
     * a mesh object was deleted, remove any enqueued operations
     * that refer to this obj.
     */

    HRESULT removeDependentOperations(const VertexPtr vert);

    bool empty() const { return c.empty(); }

    std::size_t size() const { return c.size(); }

    /**
     * Apply all of the queued operations. The queue is empty on return.
     */
    HRESULT apply();

    float getLongCutoff() const { return longCutoff;};

    float getShortCutoff() const { return shortCutoff; };

    void setShortCutoff(float);

    void setLongCutoff(float);

    ~MeshOperations();





private:

    MeshPtr mesh;

    // TODO: this would be a lot more efficient if we stack allocated
    // the ops. Can't do yet, that because of different size for each op,
    // need to fix in future versions.
    typedef std::vector<MeshOperation*> Container;

    Container c;

    void push(MeshOperation* x);


    MeshOperation* pop();

    float shortCutoff;
    float longCutoff;

    Container::iterator findDependentOperation(Container::iterator start, const TrianglePtr);

    Container::iterator findDependentOperation(Container::iterator start, const VertexPtr);

    MeshOperation *findMatchingOperation(const Edge& edge);

};

#endif /* SRC_MESHOPERATIONS_H_ */
