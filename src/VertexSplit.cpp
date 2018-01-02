/*
 * VertexSplit.cpp
 *
 *  Created on: Dec 15, 2017
 *      Author: andy
 */

#include <VertexSplit.h>
#include "MeshRelationships.h"
#include "MxMesh.h"
#include "DifferentialGeometry.h"
#include <set>
#include <limits>

typedef std::set<CCellPtr> CellSet;



/**
 * A vertex with three or more cell valences is either a skeletal vertex, or a
 * cone vertex.
 */


/**
 * Splits a vertex in the direction of the cell.
 */
static HRESULT splitVertex(MeshPtr, VertexPtr, CCellPtr);

static bool isSkeletalVertex(CVertexPtr v);

//static bool isSkeletalVertex(CVertexPtr v, VertexPtr &next, CellSet &cells);

/**
 * A cone vertex is the vertex that has the highest cell valence count
 * out of a all of it's neighboring vertices on a cell surface.
 *
 * E.g. all vertices that are form part of the given cell's surface
 * that are adjacent to a cone vertex have a lower cell valence count.
 */
static bool isConeVertex(CVertexPtr v, CCellPtr c);

/**
 * A skeletal vertex forms part of an edge that bounds two or more cells.
 */

static VertexPtr skeletalVertexFromCone(CVertexPtr cone, CellSet &cells);

static VertexPtr nextSkeletalVertex(CVertexPtr v, CVertexPtr prev,
                                    const CellSet &cells);

/**
 * Find the cell that exerts the most fore on this vertex.
 */
static CCellPtr maxForceCell(CVertexPtr);

static TrianglePtr splitEdge(MeshPtr mesh, CCellPtr cell, VertexPtr center,
        TrianglePtr t0, TrianglePtr t1, CellPtr c0, CellPtr c1, VertexPtr newVertex,
        TrianglePtr tp);



static int vertexSplitId = 0;
/**
 * Locating the next and previous skeletal vertices.
 *
 * Definition: a skeletal edge is an edge that's shared between three or more cells.
 */
VertexSplit::VertexSplit(MeshPtr _mesh, VertexPtr _vertex) :
    MeshOperation(_mesh), vertex{_vertex}, id{vertexSplitId++}
{
}


MeshOperation *VertexSplit::create(MeshPtr mesh, VertexPtr v)
{
    if (vertexSplitId > 0) return nullptr;
    
    int size = v->cells().size();

    if(size > 4 || (size > 3 && isSkeletalVertex(v))) {
        return new VertexSplit(mesh, v);
    }

    return nullptr;
}

static int cnt = 0;

HRESULT VertexSplit::apply()
{
    int size = vertex->cells().size();
    
#ifndef NDEBUG
    cnt++;
    std::vector<CellPtr> cells = vertex->cells();
#endif

    if(size <= 3 || (size <= 3 && !isSkeletalVertex(vertex))) {
        return S_OK;
    }

    // find the cell with the max force
    CCellPtr cell = maxForceCell(vertex);
    
    HRESULT result = splitVertex(mesh, vertex, cell);
    //HRESULT result = S_OK;
    
#ifndef NDEBUG
    for(CellPtr cell : cells) {
        assert(cell->isValid());
    }
#endif
    
    return result;
}

float VertexSplit::energy() const
{
    float valence = vertex->cells().size();
    return valence <= 3 ? 0. : -500 * valence;
}

bool VertexSplit::depends(CTrianglePtr tri) const
{
    return incident(tri, vertex);
}

bool VertexSplit::depends(CVertexPtr v) const
{
    return vertex == v;
}

bool VertexSplit::equals(const Edge& e) const
{
    return false;
}

void VertexSplit::mark() const
{
}


/**
 * Splits a vertex in the direction of the cell.
 *
 * Tasks
 *    1) locate
 */
static HRESULT splitVertex(MeshPtr mesh, VertexPtr center, CCellPtr cell) {

    // fan by definition must have at least three triangles
    std::vector<TrianglePtr> fan = triangleFan(center, cell);
    
#ifndef NDEBUG
    for(int i = 0; i < fan.size()-1; ++i) {
        TrianglePtr t0 = fan[i];
        TrianglePtr t1 = fan[i+1];
        assert(incident(t0, center));
        assert(incident(t1, center));
        assert(adjacent_triangle_pointers(t0, t1));
        assert(adjacent_triangle_pointers(t0, t1));
    }
#endif

    TrianglePtr firstNewTri = nullptr;
    TrianglePtr tri = nullptr;
    // which side of the first tri to connect to the last.
    CCellPtr firstTriConnectCell = nullptr;

    // new vertex is at centroid of fan
    // TODO prob better at mean pos between center and centroid.
    Vector3 fanCentroid = centroidTriangleFan(center, fan);
    VertexPtr newVertex = mesh->createVertex(((3. * center->position) + fanCentroid) / 4.);

    for(int i = 0; i < fan.size(); ++i) {
        TrianglePtr t0 = fan[i];
        TrianglePtr t1 = fan[(i+1)%fan.size()];
        
        assert(incident(t0, center));
        assert(incident(t1, center));
        assert(adjacent_triangle_pointers(t0, t1));
        assert(adjacent_triangle_pointers(t0, t1));

        CellPtr c0 = t0->cells[0] == cell ? t0->cells[1] : t0->cells[0];
        CellPtr c1 = t1->cells[0] == cell ? t1->cells[1] : t1->cells[0];

        if(c0 != c1) {
            // splitEdge(MeshPtr mesh, CCellPtr cell, VertexPtr center, TrianglePtr t0,
            // TrianglePtr t1, CellPtr c0, CellPtr c1, VertexPtr newVertex, TrianglePtr tp);
            tri = splitEdge(mesh, cell, center, t0, t1, c0, c1, newVertex, tri);

            if(!firstNewTri) {
                firstNewTri = tri;
                firstTriConnectCell = c0;
            }
        }
    }

    for(TrianglePtr tri : fan) {
        replaceTriangleVertex(tri, center, newVertex);
    }

    connect_triangle_partial_triangles(firstNewTri, tri, firstTriConnectCell);

#ifndef NDEBUG
    for(TrianglePtr tri : fan) {
        assert(tri->isValid());
    }
#endif

    return S_OK;
}

/**
 * Determine if a vertex is part of a skeletal edge.
 *
 * Vertex is skeletal if there is an edge to another vertex with the
 * same set of incident cells.
 */
static bool isSkeletalVertex(CVertexPtr v0) {
    CellSet cells{v0->cells().begin(), v0->cells().end()};

    auto sameCells = [&cells](CVertexPtr v) -> bool {
        if(cells.size() != v->cells().size()) {
            return false;
        }

        for(CellPtr cell : v->cells()) {
            if(cells.find(cell) == cells.end()) {
                return false;
            }
        }
        
        return true;
    };
    
    for(TrianglePtr tri : v0->triangles()) {
        
        for(int i = 0; i < 3; ++i) {
            if(tri->vertices[0] != v0) {
                if(sameCells(tri->vertices[i])) {
                    return true;
                }
            }
        }
    }

    return false;
}



CCellPtr maxForceCell(CVertexPtr v)
{
    float maxDiv = -std::numeric_limits<float>::max();
    float sumDiv = 0;
    CCellPtr cell = nullptr;
    for(CCellPtr c : v->cells()) {
        float d = forceDivergenceForCell(v, c);
        if(d > maxDiv) {
            maxDiv = d;
            sumDiv += d;
            cell = c;
        }
    }

    if(sumDiv == 0) {
        cell = nullptr;
        float maxCurv = -std::numeric_limits<float>::max();
        for(CCellPtr c : v->cells()) {
            float meanCurv = 0;
            float gaussCurv = 0;
            discreteCurvature(c, v, &meanCurv, &gaussCurv);
            if(meanCurv > maxCurv) {
                maxCurv = meanCurv;
                cell = c;
            }
        }

    }
    return cell;
}

bool VertexSplit::equals(CVertexPtr v) const
{
    return vertex == v;
}


/**
 * The t0 and t1 triangles correspond to the c0 and c1 cell sections.
 * @param tp: previous triangle
 */
static TrianglePtr splitEdge(MeshPtr mesh, CCellPtr cell, VertexPtr centerVertex,
        TrianglePtr t0, TrianglePtr t1, CellPtr c0, CellPtr c1,
        VertexPtr newVertex, TrianglePtr tp)
{
    // the frontier vertex, incident to both triangles
    VertexPtr frontierVertex = nullptr;
    for(VertexPtr v : t0->vertices) {
        if(v != centerVertex && incident(v, t1)) {
            frontierVertex = v;
            break;
        }
    }
    assert(frontierVertex);
    
    // index of cells in triangles
    int ci0 = t0->cellIndex(c0);
    int ci1 = t1->cellIndex(c1);
    assert(ci0 >= 0 && ci1 >= 0);
    
    // index of the middle triangle for each of the two triangles
    int t0_ai = t0->adjacentEdgeIndex(frontierVertex, centerVertex);
    int t1_ai = t1->adjacentEdgeIndex(frontierVertex, centerVertex);
    assert(t0_ai >= 0 && t1_ai >= 0);

    // find the adjacent triangles that the fan triangles connect to on the
    // opposite cell sides.
    TrianglePtr ta0 = t0->partialTriangles[ci0].neighbors[t0_ai]->triangle;
    TrianglePtr ta1 = t1->partialTriangles[ci1].neighbors[t1_ai]->triangle;
    assert(incident(ta0, frontierVertex));
    assert(incident(ta0, centerVertex));
    assert(incident(ta1, frontierVertex));
    assert(incident(ta1, centerVertex));
    
    // index of cells in the ta triangles
    int ta0_ci = ta0->cellIndex(c0);
    int ta1_ci = ta1->cellIndex(c1);
    assert(ta0_ci >= 0 && ta1_ci >= 0);

    // index of the t0 and t1 triangles from the ta triangle's perspective.
    int ta0_ti = ta0->adjacentEdgeIndex(frontierVertex, centerVertex);
    int ta1_ti = ta1->adjacentEdgeIndex(frontierVertex, centerVertex);
    assert(ta0_ti >= 0 && ta1_ti >= 0);

    assert(ta0->partialTriangles[ta0_ci].neighbors[ta0_ti]->triangle == t0);
    assert(ta1->partialTriangles[ta1_ci].neighbors[ta1_ti]->triangle == t1);

    TrianglePtr newTriangle = mesh->createTriangle({{c0, c1}},
            {{newVertex, centerVertex, frontierVertex}});

    // the center / frontier edge connects with the ta triangles, these are on the
    // staying cell, and the newVertex/frontier edge connects with the t0/t1 triangles
    // on the pulling cell side.
    // The c0 side of the triangle joins the t0 and ta0 triangles, and the c1 side
    // joins the t1 and ta1 triangles.
    int tn_ti = newTriangle->adjacentEdgeIndex(centerVertex, frontierVertex);
    int tn_ta = newTriangle->adjacentEdgeIndex(newVertex, frontierVertex);
    assert(tn_ti >= 0 && tn_ta >= 0);
    
    t0->partialTriangles[ci0].neighbors[t0_ai] = &newTriangle->partialTriangles[0];
    newTriangle->partialTriangles[0].neighbors[tn_ti] = &t0->partialTriangles[ci0];

    ta0->partialTriangles[ta0_ci].neighbors[ta0_ti] = &newTriangle->partialTriangles[0];
    newTriangle->partialTriangles[0].neighbors[tn_ta] = &ta0->partialTriangles[ta0_ci];

    t1->partialTriangles[ci1].neighbors[t1_ai] = &newTriangle->partialTriangles[1];
    newTriangle->partialTriangles[1].neighbors[tn_ti] = &t0->partialTriangles[ci1];

    ta1->partialTriangles[ta1_ci].neighbors[ta1_ti] = &newTriangle->partialTriangles[1];
    newTriangle->partialTriangles[1].neighbors[tn_ta] = &ta1->partialTriangles[ta1_ci];

    for(int i = 0; i < 2; ++i) {
        newTriangle->cells[i]->boundary.push_back(&newTriangle->partialTriangles[i]);
        if(newTriangle->cells[i]->renderer) {
            newTriangle->cells[i]->renderer->invalidate();
        }
    }

    //disconnect_triangle_vertex(t0, centerVertex);
    //disconnect_triangle_vertex(t1, centerVertex);
    //connect_triangle_vertex(t0, newVertex);
    //connect_triangle_vertex(t1, newVertex);

    if(tp) {
        connect_triangle_partial_triangles(newTriangle, tp, c0);
    }
    
    return newTriangle;
}
