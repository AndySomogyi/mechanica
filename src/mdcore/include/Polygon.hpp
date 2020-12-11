/*
 * Polygon.h
 *
 *  Created on: Dec 10, 2020
 *      Author: andy
 */

#ifndef SRC_MDCORE_SRC_POLYGON_H_
#define SRC_MDCORE_SRC_POLYGON_H_

#include <Vertex.hpp>
#include <array>


struct PartialPolygon
{
    // id of the polygon this partial belongs to.
    int32_t polygon_id;
};

struct PartialPolygonHandle : PyObject
{
    int32_t id;
};

struct Polygon
{
//    static bool classof(const CObject *o) {
//        return o->ob_type == MxPolygon_Type;
//    }
//
//    static CType *type() {return MxPolygon_Type;};
//
//    const uint id;
//
    /**
     * The triangle aspect ratio for the three corner vertex positions of a triangle.
     */
    float aspectRatio = 0;
//
//    /**
//     * Area of this triangle, calculated in positionsChanged().
//     *
//     * non-normalized normal vector is normal * area.
//     */
//    float area = 0.;
//
//    /**
//     * Normalized normal vector (magnitude is triangle area), oriented away from
//     * cellIds[0].
//     *
//     * If a cell has cellIds[0], then the normal points in the correct direction, but
//     * if the cell has cellIds[1], then the normal needs to be multiplied by -1 to point
//     * point in the correct direction.
//     */
//    Vector3 normal;
//
//    /**
//     * Geometric center (barycenter) of the triangle, computed in positionsChanged()
//     *
//     * We presently treat polygons as homogeneous patches of material, so centroid
//     * is the same as center of mass.
//     */
//    Vector3 centroid;
//
    /**
     * Polygons are stored in CCW winding order, from the zero indexed cell
     * perspective.
     *
     * indices in particles list.
     */
    std::vector<int32_t> vertices;

    /**
     * Need to associate this triangle with the cells on both sides. Trans-cell flux
     * is very frequently calculated, so optimize structure layout for both
     * trans-cell and trans-partial-triangle fluxes.
     */
    std::array<int32_t, 2> cells = {-1, -1};

    /**
     * pointers to the three triangles or edges that this triangle connects with.
     *
     * A neighbor can be either another triangle if the neighbor lies on a manifold
     * surface, or may be a skeletal edge if the lies at the intersection of three
     * cells. Currently, we restrict edges to three cells.
     */
    std::vector<int32_t> edges;


    /**
     * indices of the two partial triangles that are attached to this triangle.
     * The mesh contains a set of partial triangles.
     *
     * partialTriangles[0] contains the partial triangle for cells[0]
     */
    std::array<int32_t, 2> partialPolygons;
//
//
//    /**
//     * calculate the volume contribution this polygon has for the given cell.
//     *
//     * result is undefined if cell is not attached to this polygon.
//     */
//    float volume(CCellPtr cell) const;
//
//
//    /**
//     * Get the vertex normal for the i'th vertex in this polygon.
//     */
//    Vector3 vertexNormal(uint i, CCellPtr cell) const;
//
//
//    float vertexArea(uint i) const {
//        return _vertexAreas[i];
//    }
//
//    /**
//     * Orient the normal in the correct direction for the given cell.
//     */
//    inline Vector3 cellNormal(CCellPtr cell) const {
//        assert(cell == cells[0] || cell == cells[1]);
//        float dir = cell == cells[0] ? 1.0 : -1.0;
//        return dir * normal;
//    }
//
//    /**
//     * This is designed to heap allocated.
//     *
//     * Later versions will investigate stack allocation.
//     */
//    MxPolygon() :
//    vertices{{nullptr, nullptr, nullptr}},
//    cells{{nullptr,nullptr}},
//    partialPolygons {{
//        {nullptr, nullptr, 0.0, nullptr},
//        {nullptr, nullptr, 0.0, nullptr}
//    }},
//    id{0}
//    {}
//
//
//    MxPolygon(uint _id, CType *type);
//
//
//    inline int cellIndex(CCellPtr cell) const {
//        return (cells[0] == cell) ? 0 : ((cells[1] == cell) ? 1 : -1);
//    }
//
//    /**
//     * get the index of the given vertex in this polygons list of vertices.
//     * returns -1 if the vertex does not exist in this polygon.
//     */
//    int vertexIndex(CVertexPtr vert) const;
//
//    /**
//     * find the index of the given edge in the polygons list of edges.
//     * return -1 if the edge is not found.
//     */
//    int edgeIndex(CEdgePtr e) const;
//
//    /**
//     * get the number of sides this polygon has, equivalently, the number of
//     * vertices or edges.
//     */
//    inline uint size() const {
//        return vertices.size();
//    }
//
//    /**
//     * Inform the cell that the vertex positions have changed. Causes the
//     * cell to recalculate area and volume, also inform all contained objects.
//     */
//    HRESULT positionsChanged();
//
//    bool isConnected() const;
//
//    bool isValid() const;
//
//    bool checkEdges() const ;
//
//    float alpha = 0.5;
//
//    Magnum::Color4 color = Magnum::Color4{0.0f, 0.0f, 0.0f, 0.0f};
//
//    inline float getMass() const { return partialPolygons[0].mass + partialPolygons[1].mass; };
//
//    void setMassForCell(float val, CellPtr cell) {
//        assert(cell == cells[0] || cell == cells[1]);
//        uint cellId = cell == cells[0] ? 0 : 1;
//        partialPolygons[cellId].mass = val;
//    }
//
//private:
//    float _volume = 0.f;
//    std::vector<Vector3> _vertexNormals;
//    std::vector<float> _vertexAreas;
//
//    friend HRESULT connectPolygonVertices(MeshPtr mesh, PolygonPtr poly,
//                                          const std::vector<VertexPtr> &vertices);
//
//    friend HRESULT insertPolygonEdge(PolygonPtr poly, EdgePtr edge);
//
//    friend HRESULT disconnectPolygonEdgeVertex(PolygonPtr poly, EdgePtr edge, CVertexPtr v,
//                                               EdgePtr *e1, EdgePtr *e2);
//
//    friend HRESULT replacePolygonEdgeAndVerticesWithVertex(PolygonPtr poly, EdgePtr edge,
//                                                           VertexPtr newVert, EdgePtr* prevEdge, EdgePtr* nextEdge);
//
//    friend HRESULT Mx_SplitPolygonBisectPlane(MeshPtr mesh, PolygonPtr poly,
//                                              Vector3* normal, PolygonPtr* pn1, PolygonPtr* pn2);
//
//    friend HRESULT splitPolygonEdge(PolygonPtr poly, EdgePtr e, EdgePtr en);
//
//    friend HRESULT replacePolygonVertexWithEdgeAndVertices(PolygonPtr poly, CVertexPtr vert,
//                                                           CEdgePtr e0, CEdgePtr e1,  EdgePtr edge, VertexPtr v0, VertexPtr v1);
};

struct PolygonHandle : PyObject
{
    int32_t id;
};

/**
 */
CAPI_DATA(PyTypeObject) PartialPolygon_Type;
CAPI_DATA(PyTypeObject) Polygon_Type;

HRESULT _polygon_init(PyObject *m);

#endif /* SRC_MDCORE_SRC_POLYGON_H_ */
