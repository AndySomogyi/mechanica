/*
 * MxTriangle.h
 *
 *  Created on: Oct 3, 2017
 *      Author: andy
 */

#ifndef SRC_MXPOLYGON_H_
#define SRC_MXPOLYGON_H_

#include "MxMeshCore.h"
#include "Magnum/Math/Color.h"
#include <iostream>

enum struct Orientation {
    Inward, Outward, InwardOutward, OutwardInward, Invalid
};


struct MxPartialPolygonType : MxType {


    /**
     * Store the stoichiometry matrix in the type, initially Mechanica will
     * not support time-dependent stochiometries.
     */
};

MxAPI_DATA(MxPartialPolygonType*) MxPartialPolygon_Type;

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
struct MxPartialPolygon : MxObject {

    MxPartialPolygon(MxPartialPolygonType *type, MxPolygon *ti,
            float mass = 0, MxReal *scalars = nullptr) :
                MxObject{type}, polygon{ti},
                mass{mass}, scalarFields{scalars} {};

    /**
     * index of the triangle that this partial triangle references.
     */
    PolygonPtr polygon;


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
    float mass = 0;

    /**
     * A contiguous sequence of scalar attributes, who's time evolution is
     * defined by reactions and odes.
     */
    MxReal *scalarFields;

    std::array<float, 3> vertexAttr;



    bool isValid() const;
};


struct MxPolygonType : MxType {

};

MxAPI_DATA(MxPolygonType*) MxPolygon_Type;


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
struct MxPolygon : MxObject {

    const uint id;

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
     *
     * We presently treat polygons as homogeneous patches of material, so centroid
     * is the same as center of mass.
     */
    Vector3 centroid;

    /**
     * Polygons are stored in CCW winding order, from the zero indexed cell
     * perspective.
     */
    std::vector<VertexPtr> vertices;

    /**
     * Need to associate this triangle with the cells on both sides. Trans-cell flux
     * is very frequently calculated, so optimize structure layout for both
     * trans-cell and trans-partial-triangle fluxes.
     */
    std::array<CellPtr, 2> cells = {{nullptr}};

    /**
     * pointers to the three triangles or edges that this triangle connects with.
     *
     * A neighbor can be either another triangle if the neighbor lies on a manifold
     * surface, or may be a skeletal edge if the lies at the intersection of three
     * cells. Currently, we restrict edges to three cells.
     */
    std::vector<MxObject*> edges;

    uint edgeCount() const {
        return edges.size();
    }

    /**
     * indices of the two partial triangles that are attached to this triangle.
     * The mesh contains a set of partial triangles.
     *
     * partialTriangles[0] contains the partial triangle for cells[0]
     */
    std::array<MxPartialPolygon, 2> partialPolygons;


    /**
     * calculate the volume contribution this polygon has for the given cell.
     *
     * result is undefined if cell is not attached to this polygon.
     */
    float volume(CCellPtr cell) const;


    /**
     * Get the vertex normal for the i'th vertex in this polygon.
     */
    Vector3 vertexNormal(uint i, CCellPtr cell) const;


    float vertexArea(uint i) const {
        return _vertexAreas[i];
    }

    /**
     * Orient the normal in the correct direction for the given cell.
     */
    inline Vector3 cellNormal(CCellPtr cell) const {
        assert(cell == cells[0] || cell == cells[1]);
        float dir = cell == cells[0] ? 1.0 : -1.0;
        return dir * normal;
    }

    /**
     * This is designed to heap allocated.
     *
     * Later versions will investigate stack allocation.
     */
    MxPolygon() :
        vertices{{nullptr, nullptr, nullptr}},
        cells{{nullptr,nullptr}},
        partialPolygons {{
            {nullptr, nullptr, 0.0, nullptr},
            {nullptr, nullptr, 0.0, nullptr}
        }},
        id{0}
    {}


    MxPolygon(uint _id, MxPolygonType *type);


    inline int cellIndex(CCellPtr cell) const {
        return (cells[0] == cell) ? 0 : ((cells[1] == cell) ? 1 : -1);
    }

    inline int vertexIndex(CVertexPtr vert) const {
        for(int i = 0; i < 3; ++i) {
            if(vertices[i] == vert) {
                return i;
            }
        }
        return -1;
    }

    /**
     * get the number of sides this polygon has, equivalently, the number of
     * vertices or edges.
     */
    inline uint sides() const {
        return vertices.size();
    }

    /**
     * Inform the cell that the vertex positions have changed. Causes the
     * cell to recalculate area and volume, also inform all contained objects.
     */
    HRESULT positionsChanged();

    bool isConnected() const;

    bool isValid() const;

    float alpha = 0.5;

    Magnum::Color4 color = Magnum::Color4{0.0f, 0.0f, 0.0f, 0.0f};

    inline float getMass() const { return partialPolygons[0].mass + partialPolygons[1].mass; };

    void setMassForCell(float val, CellPtr cell) {
        assert(cell == cells[0] || cell == cells[1]);
        uint cellId = cell == cells[0] ? 0 : 1;
        partialPolygons[cellId].mass = val;
    }

private:
    float _volume = 0.f;
    std::vector<Vector3> _vertexNormals;
    std::vector<float> _vertexAreas;

    friend HRESULT connectPolygonVertices(MeshPtr mesh, PolygonPtr poly,
            const std::vector<VertexPtr> &vertices);

    friend HRESULT insertEdgeVertexIntoPolygon(EdgePtr edge, VertexPtr vert,
            PolygonPtr poly, CVertexPtr ref);
};

std::ostream& operator<<(std::ostream& os, CPolygonPtr tri);




namespace Magnum { namespace Math {

    /**
     * calculates the normal vector for three triangle vertices.
     *
     * Assumes CCW winding.
     *
     * TODO: have some global to set CCW winding.
     */
    inline Vector3<float> triangleNormal(const Vector3<float>& v1,
            const Vector3<float>& v2, const Vector3<float>& v3) {
        return Magnum::Math::cross(v2 - v1, v3 - v1);
    }

    inline Vector3<float> triangleNormal(const std::array<Vector3<float>, 3> &verts) {
        return triangleNormal(verts[0], verts[1], verts[2]);
    }

    inline Vector3<float> triangleNormal(const std::array<VertexPtr, 3> &verts) {
        return triangleNormal(verts[0]->position, verts[1]->position, verts[2]->position);
    }

    // Nx = UyVz - UzVy
    // Ny = UzVx - UxVz
    // Nz = UxVy - UyVx
    // non-normalized normal vector
    // multiply by neg 1, CCW winding.


    inline float triangleArea(const Vector3<float>& v1,
            const Vector3<float>& v2, const Vector3<float>& v3) {
        Vector3<float> abnormal = Math::triangleNormal(v1, v2, v3);
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

#endif /* SRC_MXPOLYGON_H_ */
