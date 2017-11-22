/*
 * MxTriangle.h
 *
 *  Created on: Oct 3, 2017
 *      Author: andy
 */

#ifndef SRC_MXTRIANGLE_H_
#define SRC_MXTRIANGLE_H_

#include "MxMeshCore.h"


struct MxPartialTriangleType : MxType {


    /**
     * Store the stoichiometry matrix in the type, initially Mechanica will
     * not support time-dependent stochiometries.
     */
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
 */
struct MxPartialTriangle : MxObject {

    MxPartialTriangle(MxPartialTriangleType *type, MxTriangle *ti,
            const PartialTriangles& neighbors = {{nullptr, nullptr, nullptr}},
            float mass = 0, MxReal *scalars = nullptr) :
                MxObject{type}, triangle{ti}, neighbors{neighbors},
                mass{mass}, scalarFields{scalars} {};

    /**
     * index of the triangle that this partial triangle references.
     */
    TrianglePtr triangle;

    /**
     * indices of the three neighboring partial triangles.
     */
    std::array<MxPartialTriangle*, 3> neighbors;


    float mass;

    /**
     * A contiguous sequence of scalar attributes, who's time evolution is
     * defined by reactions and odes.
     */
    MxReal *scalarFields;

    int unboundNeighborCount() {
        int count = 0;
        for(int i = 0; i < 3; ++i) {
            if(neighbors[i] == nullptr) {
                count += 1;
            }
        }
        return count;
    }
};


struct MxTriangleType : MxType {

};


/**
 * Represents a triangle, connected to two partial triangles.
 *
 * One of the primary tasks of the Mechanica simulation environment
 * is to calculate trans-cell fluxes. Hence, the MxTriangle provides a
 * direct way to get at the cells on either side.
 *
 * The MxPartialTriangles represent a section of material surface, so,
 * only partial triangles have attached attributes. The triangle (here)
 * only connects partial triangles.
 *
 * Tasks:
 *     * provide connection between adjacent cells so that we can calculate
 *       fluxes
 *     * represent the geometry of a triangle.
 *     *
 */
struct MxTriangle : MxObject {

    /**
     * The triangle aspect ratio for the three corner vertex positions of a triangle.
     */
    float aspectRatio = 0;

    /**
     * Area of this triangle, calculated in positionsChanged().
     *
     * non-normalized normal vector is normal * area.
     */
    float area = 0.;

    /**
     * The triangle is the fundamental material unit in Mechanica. All material
     * properties are associated with the triangle. The vertex mass / area
     * values are determined FROM the triangle. Only set the triangle mass,
     * as the triangle will calculate the vertex mass from the triangle mass.
     *
     * The mass of a triangle can vary, depending on the density and amount
     * of attached scalar quantities. Mass is the weighted average of the
     * attached scalar quantity densities times the area.
     */
    float mass = 0.;

    /**
     * Normalized normal vector (magnitude is triangle area), oriented away from
     * cellIds[0].
     *
     * If a cell has cellIds[0], then the normal points in the correct direction, but
     * if the cell has cellIds[1], then the normal needs to be multiplied by -1 to point
     * point in the correct direction.
     */
    Vector3 normal;

    /**
     * Geometric center (barycenter) of the triangle, computed in positionsChanged()
     */
    Vector3 centroid;

    /**
     * Each triangle belongs to exactly one facet.
     */
    FacetPtr facet = nullptr;

    /**
     * indices of the 3 vertices in the MxMesh that make up this partial face,
     * in the correct winding order. The winding of these vertices correspond to the
     * normal vector.
     */
    std::array<VertexPtr, 3> vertices;

    /**
     * Need to associate this triangle with the cells on both sides. Trans-cell flux
     * is very frequently calculated, so optimize structure layout for both
     * trans-cell and trans-partial-triangle fluxes.
     */
    std::array<CellPtr, 2> cells;

    /**
     * pointers to the three triangles around the ring edges.
     */
    std::array<TrianglePtr, 3> edgeRing;

    /**
     * indices of the two partial triangles that are attached to this triangle.
     * The mesh contains a set of partial triangles.
     *
     * partialTriangles[0] contains the partial triangle for cells[0]
     */
    std::array<MxPartialTriangle, 2> partialTriangles;

    uint32_t id;

    /**
     * does this triangle match the given set of vertex
     * indices.
     *
     * There are 3 possible forward combinations of 3 vertices,
     * {1,2,3}, {2,3,1}, {3,1,2}, if any of these combinations
     * match, then this returns a 1. Similarly, there are 3 possible
     * reversed combinations, {3,2,1}, {2,1,3}, {1,3,2}, if any of these
     * match, then method returns a -1, otherwise, 0.
     *
     * @returns -1 if the vertices match in reverse order, or a
     *             reversed permutation.
     *
     * @returns 0 if the is no match
     * @returns 1 if the vertices match in the same order
     */
    int matchVertexIndices(const std::array<VertexPtr, 3> &vertInd);


    /**
     * Orient the normal in the correct direction for the given cell.
     */
    inline Vector3 cellNormal(const CellPtr cell) const {
        assert(cell == cells[0] || cell == cells[1]);
        float dir = cell == cells[0] ? 1.0 : -1.0;
        return dir * normal;
    }

    /**
     * This is designed to heap allocated.
     *
     * Later versions will investigate stack allocation.
     */
    MxTriangle() :
        vertices{{nullptr, nullptr, nullptr}},
        cells{{nullptr,nullptr}},
        partialTriangles {{
            {nullptr, nullptr, {{nullptr, nullptr, nullptr}}, 0.0, nullptr},
            {nullptr, nullptr, {{nullptr, nullptr, nullptr}}, 0.0, nullptr}
        }},
        id{0}
    {}


    MxTriangle(MxTriangleType *type, const std::array<VertexPtr, 3> &vertices,
            const std::array<CellPtr, 2> &cells = {{nullptr, nullptr}},
            const std::array<MxPartialTriangleType*, 2> &partialTriangleTypes = {{nullptr, nullptr}},
            FacetPtr facet = nullptr);


    /**
     * get the edge index of edge from a pair of vertices.
     *
     * If vertices[0] == a and vertices[1] == b or vertices[1] == a and vertices[0] == b,
     * then the edge is in the zero position, and so forth.
     *
     * If the edge does not match, then returns a -1.
     */
    int adjacentEdgeIndex(const VertexPtr a, const VertexPtr b) const;



    /**
     * Disconnects a corner of this triangle from oldVertex, and re-attaches
     * this triangle to newVertex.
     */
    VertexPtr replaceChild(VertexPtr newVertex, VertexPtr oldVertex);



    /**
     * Inform the cell that the vertex positions have changed. Causes the
     * cell to recalculate area and volume, also inform all contained objects.
     */
    HRESULT positionsChanged();

    bool isConnected();


    bool isValid();

    float alpha = 0.5;
};

namespace Magnum { namespace Math {

    /**
     * calculates the normal vector for three triangle vertices.
     *
     * Assumes CCW winding.
     *
     * TODO: have some global to set CCW winding.
     */
    inline Vector3<float> normal(const Vector3<float>& v1,
            const Vector3<float>& v2, const Vector3<float>& v3) {
        return Magnum::Math::cross(v2 - v1, v3 - v1);
    }

    // Nx = UyVz - UzVy
    // Ny = UzVx - UxVz
    // Nz = UxVy - UyVx
    // non-normalized normal vector
    // multiply by neg 1, CCW winding.


    inline float triangle_area(const Vector3<float>& v1,
            const Vector3<float>& v2, const Vector3<float>& v3) {
        Vector3<float> abnormal = Math::normal(v1, v2, v3);
        float len = abnormal.length();
        return 0.5 * len;
    }

    inline float distance(const Vector3<float>& a, const Vector3<float>& b) {
        return (a - b).length();
    }

    inline float distance_sqr(const Vector3<float>& a, const Vector3<float>& b) {
        Vector3<float> diff = a - b;
        return dot(diff, diff);
    }
}}

#endif /* SRC_MXTRIANGLE_H_ */
