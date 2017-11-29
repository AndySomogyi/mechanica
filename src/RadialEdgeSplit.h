/*
 * RadialEdgeSplit.h
 *
 *  Created on: Nov 23, 2017
 *      Author: andy
 */

#ifndef SRC_RADIALEDGESPLIT_H_
#define SRC_RADIALEDGESPLIT_H_

#include "MeshOperations.h"



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

    virtual void mark() const;

private:

    std::default_random_engine randEngine;
    std::uniform_real_distribution<float> uniformDist;

    float longCutoff;
    Edge edge;
};

#endif /* SRC_RADIALEDGESPLIT_H_ */
