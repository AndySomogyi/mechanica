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

struct IncidentCell {
    CellPtr cell;
    int neighborCount;
    float curvature;
};

static std::vector<IncidentCell> incidentCells(CVertexPtr vertex);


/**
 * A vertex with three or more cell valences is either a skeletal vertex, or a
 * cone vertex.
 */


/**
 * Splits a vertex in the direction of the cell.
 */
static HRESULT splitVertex(MeshPtr, VertexPtr, CellPtr);

/**
 * determine the largest shared cell count this vertex has with any
 * of it's neighbors.
 */
static int sharedCellCount(CVertexPtr v);

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

/**
 * looks at all the triangles that the given cell has that are incident to the vertex,
 * and counts how many faces the cell shares with the other cells that are also
 * incident to the vertex.
 */
static int cellNeighborCount(const CellSet &cells, CCellPtr cell, CVertexPtr vertex);



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

static int nonRootCellCount(CVertexPtr v) {
    int cnt = 0;
    for(CCellPtr c : v->cells()) {
        if(!c->isRoot()) {
            cnt += 1;
        }
    }
    return cnt;
}

static CellPtr shouldSplitVertex(VertexPtr vertex);

enum FanOrdering {CW, CCW, INVALID};

static FanOrdering triangleFanOrdering(CTrianglePtr tri, CCellPtr cell, CVertexPtr centerVert, CVertexPtr fanVert) {
    int cellIndex = tri->cellIndex(cell);
    int centerIndex = tri->vertexIndex(centerVert);
    int fanIndex = tri->vertexIndex(fanVert);

    if(cellIndex == 0) {
        if((centerIndex+1)%3 == fanIndex) {
            return FanOrdering::CCW;
        }
        else if((fanIndex+1)%3 == centerIndex) {
            return FanOrdering::CW;
        }
        else {
            assert(0 && "invalid triangle ordering");
        }
    } else {
        assert(cellIndex == 1);
        if((fanIndex+1)%3 == centerIndex) {
            return FanOrdering::CCW;
        }
        else if((centerIndex+1)%3 == fanIndex) {
            return FanOrdering::CW;
        }
        else {
            assert(0 && "invalid triangle ordering");
        }
    }
    return FanOrdering::INVALID;
}


MeshOperation *VertexSplit::create(MeshPtr mesh, VertexPtr v)
{
    if (shouldSplitVertex(v)) {
        std::cout << "queued vertex: " << v << std::endl;
        
        return new VertexSplit(mesh, v);
    }
    //if (vertexSplitId > 1) return nullptr;

    /*


    int size = v->cells().size();

    if(size > 5 ||
       (size > 3 && sharedCellCount(v) > 3) ){ //||
//       (size > 3 && nonRootCellCount(v) > 3)) {
        ValenceCell lowest = lowestValenceCell(v);

        if(lowest.valenceCnt + 1 >= size) {
            std::cout << "can't split vertex, don't have a low valence count cell" << std::endl;
            return nullptr;
        }

        std::cout << "queued VertexSplit, vertex: " << v << std::endl
            << "cell count: " << size << ", " << std::endl
            << "sharedCellCount: " << sharedCellCount(v) << std::endl
            << "nonRootCellCount: " << nonRootCellCount(v) << std::endl
            << "lowestValenceCellCount: " << lowest.valenceCnt << std::endl
            << "lowestValenceCell: " << lowest.cell->id;
        return new VertexSplit(mesh, v);
    }
     */

    return nullptr;
}

static bool tracedVerifyVertex(VertexPtr vertex, bool trace)  {

    for(TrianglePtr tri : vertex->triangles()) {

        if(trace) {
            std::cout << "tri: " << tri << std::endl;
        }


        for(VertexPtr v : tri->vertices) {
            if(trace) {
                std::cout << "v: " << v << std::endl;
            }

            for(CCellPtr c : v->cells()) {
                if(trace) {
                    std::cout << "c: " << c->id << std::endl;
                }

                TrianglePtr first = v->triangleForCell(c);

                if(trace) {
                    std::cout << "first: " << first << std::endl;
                }

                // the loop triangle
                TrianglePtr t = first;
                // keep track of the previous triangle
                TrianglePtr prev = nullptr;
                do {
                    //CTrianglePtr next = tri->nextTriangleInFan(vi, cell, prev);
                    //prev = tri;
                    //tri = next;

                    //assert(prev && next);

                    if(trace) {
                        std::cout << "calling nextTriangleFan" << std::endl;
                        std::cout << "t: " << t << std::endl;
                        if(prev) {
                            std::cout << "prev: " << prev << std::endl;
                        } else {
                            std:: cout << "prev: null" << std::endl;
                        }
                    }

                    TrianglePtr next = t->nextTriangleInFan(v, c, prev);

                    if(trace) {
                        std::cout << "nextTriangleFan(v:" << v->id << ", c:" << c->id << ", prev: ";
                        if(prev) {
                            std::cout << prev->id ;
                        } else {
                            std::cout << "null";
                        }
                        std::cout << ") ->" << std::endl;
                        if(next) {
                            std::cout << next << std::endl;
                        } else {
                            std::cout << "null" << std::endl;
                        }
                    }

                    if(!next) {
                        std::cout << "error, triangle " << t << " returned null for prev tri: " << std::endl;
                        if(prev) {
                            std::cout << prev << std::endl;
                        } else {
                            std::cout << "null" << std::endl;
                        }
                        std::cout << "cell id: " << c->id << std::endl;
                        std::cout << "very bad" << std::endl;

                        return false;
                    }
                    prev = t;
                    t = next;
                    assert(prev && next);
                } while(t && t != first);
            }
        }
    }
    return true;
}

static void reconnectPathologicalPTriangles(TrianglePtr ta, int ca_i, TrianglePtr tb, int cb_i);


static void verifyVertex(VertexPtr vertex) {
    if(tracedVerifyVertex(vertex, false)) {
        return;
    }

    std::cout << "verify vertex failed, tracing..." << std::endl;
    tracedVerifyVertex(vertex, true);
}



static int cnt = 0;

HRESULT VertexSplit::apply()
{
    CellPtr splitCell = shouldSplitVertex(vertex);
    if(!splitCell) {
        return S_OK;
    }
    else {
        std::cout << "VertexSplit::apply(), vertex: " << vertex << std::endl;
    }

#if 0

    for(CellPtr cell : mesh->cells) {
        assert(cell->isValid());
    }

    for(TrianglePtr tri : mesh->triangles) {
        if(!tri->isValid()) {
            std::cout << "Bad Triangle!" << std::endl;
            std::cout << tri << std::endl;
            tri->isValid();
            //assert(0);
        }
    }
    //verifyVertex(vertex);
#endif



#ifndef NDEBUG
    cnt++;
    std::vector<CellPtr> cells = vertex->cells();
#endif



    HRESULT result = splitVertex(mesh, vertex, splitCell);

    for(TrianglePtr tri : vertex->triangles()) {
        tri->positionsChanged();
    }

    for(CellPtr cell : vertex->cells()) {
        cell->updateDerivedAttributes();
    }

    splitCell->updateDerivedAttributes();

    //HRESULT result = S_OK;

#ifndef NDEBUG
    for(CellPtr cell : mesh->cells) {
        assert(cell->isValid());
    }

    for(TrianglePtr tri : mesh->triangles) {
        if(!tri->isValid()) {
            std::cout << "Bad Triangle!" << std::endl;
            std::cout << tri << std::endl;
            tri->isValid();
            //assert(0);
        }
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
    return vertex == e[0] || vertex == e[1];
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
static HRESULT splitVertex(MeshPtr mesh, VertexPtr center, CellPtr cell) {

    // fan by definition must have at least three triangles
    std::vector<TrianglePtr> fan = triangleFan(center, cell);

    std::cout << "triangle fan" << std::endl;
    for(int i = 0; i < fan.size(); ++i) {
        std::cout << "triangle_fan[" << i << "]: " << fan[i] << std::endl;
    }

    std::vector<TrianglePtr> newTris;

    // keep track of cells, need to tell them to update thier attributes when done
    // changing geomtry.
    std::set<CellPtr> cells{cell};

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

    // new vertex, push the triangle fan in the opposite direction of it's normal.

    Vector3 fanNormal = normalTriangleFan(cell, fan);
    float distance = mesh->getShortCutoff() >= 0.0001 ? 1.5 * mesh->getShortCutoff() : 0.00015;
    VertexPtr newVertex = mesh->createVertex(center->position - distance * fanNormal);

#ifndef NDEBUG
    Vector3 centerPos = center->position - cell->centroid;
    float fanDir = Math::dot(fanNormal, centerPos);
    std::cout << "fan dir: " << fanDir << std::endl;
    std::cout << "new vertex: " << newVertex << std::endl;
#endif

    for(int i = 0; i < fan.size(); ++i) {
        TrianglePtr t0 = fan[i];
        TrianglePtr t1 = fan[(i+1)%fan.size()];

        assert(incident(t0, center));
        assert(incident(t1, center));
        assert(adjacent_triangle_pointers(t0, t1));
        assert(adjacent_triangle_pointers(t0, t1));

        int c0_i = t0->cells[0] == cell ? 1 : 0;
        int c1_i = t1->cells[0] == cell ? 1 : 0;

        CellPtr c0 = t0->cells[c0_i];
        CellPtr c1 = t1->cells[c1_i];

        cells.insert(c0);
        cells.insert(c1);

        if(c0 != c1) {
            // splitEdge(MeshPtr mesh, CCellPtr cell, VertexPtr center, TrianglePtr t0,
            // TrianglePtr t1, CellPtr c0, CellPtr c1, VertexPtr newVertex, TrianglePtr tp);
            tri = splitEdge(mesh, cell, center, t0, t1, c0, c1, newVertex, tri);

            if(!firstNewTri) {
                firstNewTri = tri;
                firstTriConnectCell = c0;
            }

            newTris.push_back(tri);
        }

        // we now know that these triangles belong to the same cell, but could have
        // a situation where they are not connected, deal with it.
        else if(!adjacent(&t0->partialTriangles[c0_i], &t1->partialTriangles[c1_i])) {
            assert(c0 == c1);
            std::cout << "same side partial triangles not adjacent" << std::endl;
            reconnectPathologicalPTriangles(t0, c0_i, t1, c1_i);
        }
    }

#ifndef NDEBUG
    for(TrianglePtr tri : fan) {
        Orientation before = tri->orientation();
        replaceTriangleVertex(tri, center, newVertex);
        tri->positionsChanged();
        if(tri->orientation() != before) {
            std::cout << "WARNING: triangle orientation in fan changed after moving: "
                      << tri << std::endl;
        }
    }
#else
    for(TrianglePtr tri : fan) {
        replaceTriangleVertex(tri, center, newVertex);
        tri->positionsChanged();
    }
#endif



    for(CellPtr cell : cells) {
        cell->updateDerivedAttributes();
    }

    if(newTris.size()) {
        assert(firstNewTri && tri && firstTriConnectCell);
        connect_triangle_partial_triangles(firstNewTri, tri, firstTriConnectCell);

        for(TrianglePtr tri : newTris) {
            tri->positionsChanged();
            assert(tri->isValid());
            //assert(tri->orientation() == Orientation::Outward);
        }
    }



#ifndef NDEBUG
    
    for(TrianglePtr tri : fan) {
        assert(tri->isValid());
    }

    /*


    for(TrianglePtr tri : mesh->triangles) {
        if(!tri->isValid()) {
            tri->isValid();

            assert(0);
        }

        for(int i = 0; i < 3; ++i) {
            EdgeTriangles et({{tri->vertices[i], tri->vertices[(i+1)%3]}});
            assert(et.isValid());
        }
    }
    */

    std::cout << "finished splitting vertex, center vertex: " << std::endl
    << center << std::endl
    << ", new vertex: " << newVertex << std::endl
    << ", new vertex cell count: " << newVertex->cells().size() << std::endl
    << ", new vertex shared cell count: " << sharedCellCount(newVertex) << std::endl
    << ", center vertex: " << center << std::endl
    << ", center vertex cell count: " << center->cells().size() << std::endl
    << ", center vertex shared cell count: " << sharedCellCount(center) << std::endl;


#endif


    return S_OK;
}

/**
 * Determine if a vertex is part of a skeletal edge.
 *
 * Vertex is skeletal if there is an edge to another vertex with the
 * same set of incident cells.
 */
static int sharedCellCount(CVertexPtr v0) {

    CellSet cells;
    for(CellPtr c : v0->cells()) {
        //if(!c->isRoot()) {
            cells.insert(c);
        //}
    }

    int maxCnt = 0;

    for(TrianglePtr tri : v0->triangles()) {
        for(VertexPtr v : tri->vertices) {
            if(v == v0) {
                continue;
            }

            int cnt = 0;
            for(CellPtr cell : v->cells()) {
                if(cells.find(cell) != cells.end()) {
                    cnt += 1;
                }
            }

            if(cnt > maxCnt) {
                maxCnt = cnt;
            }
        }
    }

    return maxCnt;
}



bool VertexSplit::equals(CVertexPtr v) const
{
    return vertex == v;
}

#ifndef NDEBUG
static bool verifyVertexOrdering(const std::array<VertexPtr, 3> &verts,
        CCellPtr c0, CCellPtr c1) {

    Vector3 triCentroid = (verts[0]->position + verts[1]->position + verts[2]->position) / 3;
    Vector3 triNormal = Math::normal(verts);
    if(!c0->isRoot() && Math::dot(triCentroid - c0->centroid, triNormal) < 0) {
        std::cout << "bad tri ordering" << std::endl;
        return false;

    }

    if(!c1->isRoot() && Math::dot(triCentroid - c1->centroid, triNormal) > 0) {
        std::cout << "bad tri ordering" << std::endl;
        return false;
    }

    return true;
}
#endif


/**
 * In a vertex split, all of the triangles attached to the pulled cell's cone
 * are centered at the center vertex. All of these triangles get disconnected from the
 * center, and re-attached to a new center that moves a little bit towards the base
 * of the cone.
 * The t0 and t1 triangles correspond to the c0 and c1 cell sections.
 *
 * These two triangles, t0 and t1 are part of the triangle fan that forms the
 * part of the cell that will be detached section that will pull away. All of the
 * triangles in this fan will originally are attached to the center vertex. Later,
 * this center vertex will be removed from all of the triangles in the fan, and
 * replaced with the new vertex.
 *
 * Returns a new triangle that is added as a T to the beween the given t0 and t1
 * triangles, this new triangle gets added to the cell.
 *
 * @param tp: previous triangle
 */
static TrianglePtr splitEdge(MeshPtr mesh, CCellPtr cell, VertexPtr centerVertex,
        TrianglePtr t0, TrianglePtr t1, CellPtr c0, CellPtr c1,
        VertexPtr newVertex, TrianglePtr tp)
{
    // the frontier vertex, incident to both triangles
    VertexPtr frontierVertex = nullptr;
    int t0_frontierIndex = -1;
    for(int i = 0; i < 3; ++i) {
        VertexPtr v = t0->vertices[i];
        if(v != centerVertex && incident(v, t1)) {
            frontierVertex = v;
            t0_frontierIndex = i;
            break;
        }
    }
    assert(frontierVertex);

    int t0_centerIndex = -1;
    for(int i = 0; i < 3; ++i) {
        if(t0->vertices[i] == centerVertex) {
            t0_centerIndex = i;
            break;
        }
    }
    assert(t0_centerIndex >= 0);

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

    // Vertex winding order is CCW, that means that the normal points out, in the
    // direction of CCW winding, following the right hand rule.
    // for the normal to piont away from the cell in the zero position,
    // need vertices would as frontier -> center -> new
    // The center vertex is the original vertex that's being split. All of the
    // triangles attached to the fan get pushed inwards towards their
    // cell center.

    std::array<VertexPtr, 3> vertices;

    // triangleFanOrdering(CTrianglePtr tri, CCellPtr cell, CVertexPtr centerVert, CVertexPtr fanVert)
    if(triangleFanOrdering(t0, c0, centerVertex, frontierVertex) == FanOrdering::CCW) {
        assert(triangleFanOrdering(t1, c1, centerVertex, frontierVertex) == FanOrdering::CW);
        vertices = {{centerVertex, frontierVertex, newVertex}};
        //assert(verifyVertexOrdering(vertices, c0, c1));
    }
    else {
        assert(triangleFanOrdering(t0, c0, centerVertex, frontierVertex) == FanOrdering::CW);
        assert(triangleFanOrdering(t1, c1, centerVertex, frontierVertex) == FanOrdering::CCW);
        vertices = {{centerVertex, newVertex, frontierVertex}};
        assert(verifyVertexOrdering(vertices, c0, c1));
    }


#ifndef NDEBUG
    if(tp) {
        Vector3 triCentroid = (newVertex->position + frontierVertex->position + centerVertex->position) / 3;
        Vector3 triPos = triCentroid - c0->centroid;
        Vector3 triNormal = Math::normal(vertices);
        float dir = Math::dot(triNormal, tp->normal);
        float cdir = Math::dot(triNormal, triPos);
        std::cout << "c0 id: " << c0->id << ", prev tri dir: " << dir << ", center dir: " << cdir << std::endl;
        std::cout << "foo" << std::endl;

    }
#endif




    //TrianglePtr newTriangle = mesh->createTriangle({{c0, c1}},
    //        {{newVertex, centerVertex, frontierVertex}});
    TrianglePtr newTriangle = mesh->createTriangle({{c0, c1}}, vertices);

    auto diff = newTriangle->centroid - c0->centroid;
    float d = Math::dot(diff, newTriangle->cellNormal(c0));

    diff = t0->centroid - c0->centroid;
    auto d2 = Math::dot(diff, t0->cellNormal(c0));

    std::cout << "dir: " << d << ", t0 dir: " << d2 << std::endl;

    /*
    std::cout << "newVert:" << newVertex << std::endl;
    std::cout << "centerVert:" << centerVertex << std::endl;
    std::cout << "frontierVert:" << frontierVertex << std::endl;
    std::cout << "c0:" << c0->id << std::endl;
    std::cout << "c1:" << c1->id << std::endl;
    std::cout << "t0: " << t0 << std::endl;
    std::cout << "t1: " << t1 << std::endl;
    std::cout << "ta0: " << ta0 << std::endl;
    std::cout << "ta1: " << ta1 << std::endl;
     */

    assert(ta0_ti >= 0 && ta1_ti >= 0);

    assert(ta0->partialTriangles[ta0_ci].neighbors[ta0_ti]->triangle == t0);
    assert(ta1->partialTriangles[ta1_ci].neighbors[ta1_ti]->triangle == t1);

    newTriangle->color = Color4{1., 1., 0., 0.5};



    // the center / frontier edge connects with the ta triangles, these are on the
    // staying cell, and the newVertex/frontier edge connects with the t0/t1 triangles
    // on the pulling cell side.
    // The c0 side of the triangle joins the t0 and ta0 triangles, and the c1 side
    // joins the t1 and ta1 triangles.
    int tn_ta = newTriangle->adjacentEdgeIndex(centerVertex, frontierVertex);
    int tn_ti = newTriangle->adjacentEdgeIndex(newVertex, frontierVertex);
    assert(tn_ti >= 0 && tn_ta >= 0);

    t0->partialTriangles[ci0].neighbors[t0_ai] = &newTriangle->partialTriangles[0];
    newTriangle->partialTriangles[0].neighbors[tn_ti] = &t0->partialTriangles[ci0];

    ta0->partialTriangles[ta0_ci].neighbors[ta0_ti] = &newTriangle->partialTriangles[0];
    newTriangle->partialTriangles[0].neighbors[tn_ta] = &ta0->partialTriangles[ta0_ci];

    t1->partialTriangles[ci1].neighbors[t1_ai] = &newTriangle->partialTriangles[1];
    newTriangle->partialTriangles[1].neighbors[tn_ti] = &t1->partialTriangles[ci1];

    ta1->partialTriangles[ta1_ci].neighbors[ta1_ti] = &newTriangle->partialTriangles[1];
    newTriangle->partialTriangles[1].neighbors[tn_ta] = &ta1->partialTriangles[ta1_ci];

    // move the mass from the new triangle's neighbors to the new triangle.
    // the existing triangles have not been moved yet, so we still know their
    // area, but we know the size of the new traiangle. Idea is we take half
    // the new triangle's mass from one neigbor, and the other half from the
    // other neighbor. We scale the mass by the area fraction of the new
    // triangle to it's neigbors.
    {
        float t0_frac  = 0.5 * newTriangle->area / (t0->area  + 0.5 * newTriangle->area);
        float t1_frac  = 0.5 * newTriangle->area / (t1->area  + 0.5 * newTriangle->area);
        float ta0_frac = 0.5 * newTriangle->area / (ta0->area + 0.5 * newTriangle->area);
        float ta1_frac = 0.5 * newTriangle->area / (ta1->area + 0.5 * newTriangle->area);

        assert(t0_frac <= 1 && t1_frac <= 1 && ta0_frac <= 1 && ta1_frac);

        newTriangle->partialTriangles[0].mass = t0_frac * t0->partialTriangles[ci0].mass;
        t0->partialTriangles[ci0].mass -= t0_frac * t0->partialTriangles[ci0].mass;

        newTriangle->partialTriangles[0].mass += ta0_frac * ta0->partialTriangles[ta0_ci].mass;
        ta0->partialTriangles[ta0_ci].mass -= ta0_frac * ta0->partialTriangles[ta0_ci].mass;

        newTriangle->partialTriangles[1].mass += t1_frac * t1->partialTriangles[ci1].mass;
        t1->partialTriangles[ci1].mass -= t1_frac * t1->partialTriangles[ci1].mass;

        newTriangle->partialTriangles[1].mass += ta1_frac * ta1->partialTriangles[ta1_ci].mass;
        ta1->partialTriangles[ta1_ci].mass -= ta1_frac * ta1->partialTriangles[ta1_ci].mass;
    }

    for(int i = 0; i < 2; ++i) {
        newTriangle->cells[i]->boundary.push_back(&newTriangle->partialTriangles[i]);
        if(newTriangle->cells[i]->renderer) {
            newTriangle->cells[i]->renderer->invalidate();
        }
    }

    if(tp) {
        connect_triangle_partial_triangles(newTriangle, tp, c0);
    }

    
    std::cout << "VertexSplit created new triangle: " << newTriangle << std::endl;

    return newTriangle;
}

static int cellNeighborCount(const CellSet &cells, CCellPtr cell, CVertexPtr vertex) {
    int count = 0;

    // get the first triangle
    TrianglePtr first = vertex->triangleForCell(cell);
    assert(first);

    // the loop triangle
    TrianglePtr tri = first;

    // keep track of the previous triangle
    TrianglePtr prev = nullptr;

    do {
        assert(tri->cells[0] == cell || tri->cells[1] == cell);
        CCellPtr otherCell = (tri->cells[0] == cell) ? tri->cells[1] : tri->cells[0];

        if(cells.find(otherCell) != cells.end()) {
            count += 1;
        }

        TrianglePtr next = tri->nextTriangleInFan(vertex, cell, prev);
        prev = tri;
        tri = next;
    } while(tri && tri != first);

    return count;
}

static std::vector<IncidentCell> incidentCells(CVertexPtr vertex) {
    std::vector<IncidentCell> incidentCells(vertex->cells().size());

    CellSet cells{vertex->cells().begin(), vertex->cells().end()};

    int i = 0;
    for(CellPtr c : vertex->cells()) {
        incidentCells[i].cell = c;
        incidentCells[i].neighborCount = cellNeighborCount(cells, c, vertex);
        incidentCells[i].curvature = umbrella(vertex, c);
        i += 1;
    }

    return incidentCells;
}

static CellPtr shouldSplitVertex(VertexPtr v) {

    const int size = v->cells().size();

    // no topological advantage to less than 4 splits.
    // a 3 valence split produces another 3 valence vertex
    if(size <= 2) {
        return nullptr;
    }

    // get all of the non-root cells
    CellSet nonRootCells;
    for(CCellPtr c : v->cells()) {
        if(!c->isRoot()) {
            nonRootCells.insert(c);
        }
    }

    CellPtr cell = nullptr;

    // count of non-root neighbors
    int neighborCount = std::numeric_limits<int>::max();

    float curvature = 0;

    for(CellPtr c : v->cells()) {
        int nCount = cellNeighborCount(nonRootCells, c, v);

        if((nCount < neighborCount)) {
            cell = c;
            neighborCount = nCount;
            curvature = umbrella(v, c);
        }
        else if(nCount == neighborCount) {
            float curv = umbrella(v, c);
            if(curv > curvature) {
                cell = c;
                neighborCount = nCount;
                curvature = curv;
            }
        }
    }

    assert(cell);

    // this is a standard skeletal edge
    if(size <= 3 && neighborCount >= 1) {
        return nullptr;
    }

    // this is an exposed edge, where we have a junction of 2 or 3 cells
    // at root facing surface, keep this.
    if(nonRootCells.size() <= 3 && neighborCount >= 2) {
        return nullptr;
    }

    // internal 4 vertex, with 4 incident cells, keep this
    if(nonRootCells.size() <= 4 && size <= 4 && neighborCount >= 2) {
        return nullptr;
    }


    std::cout << "should split vertex: " << v << std::endl
    << "cell count: " << size << ", " << std::endl
    << "sharedCellCount: " << sharedCellCount(v) << std::endl
    << "nonRootCellCount: " << nonRootCells.size() << std::endl
    << "neighborCount: " << neighborCount << std::endl
    << "split cell id: " << cell->id << std::endl
    << "curvature: " << curvature << std::endl;

    return cell;
}

static void reconnectPathologicalPTriangles(TrianglePtr ta, int ca_i, TrianglePtr tb, int cb_i) {
    int sharedCell_ai = (ca_i + 1) % 2;
    int sharedCell_bi = (cb_i + 1) % 2;
    int neighbor_ai = -1;
    int neighbor_bi = -1;

    for(int i = 0; i < 3; ++i) {
        if(ta->partialTriangles[sharedCell_ai].neighbors[i] == &tb->partialTriangles[sharedCell_bi]) {
            neighbor_ai = i;
            break;
        }
    }

    for(int i = 0; i < 3; ++i) {
        if(tb->partialTriangles[sharedCell_bi].neighbors[i] == &ta->partialTriangles[sharedCell_ai]) {
            neighbor_bi = i;
            break;
        }
    }

    assert(neighbor_ai >= 0 && neighbor_bi >= 0);

    PTrianglePtr partTri_a = ta->partialTriangles[ca_i].neighbors[neighbor_ai];
    PTrianglePtr partTri_b = tb->partialTriangles[cb_i].neighbors[neighbor_bi];

    int neighborPt_ai = -1;
    int neighborPt_bi = -1;

    for(int i = 0; i < 3; ++i) {
        if(partTri_a->neighbors[i] == &ta->partialTriangles[ca_i]) {
            neighborPt_ai = i;
            break;
        }
    }

    for(int i = 0; i < 3; ++i) {
        if(partTri_b->neighbors[i] == &tb->partialTriangles[cb_i]) {
            neighborPt_bi = i;
            break;
        }
    }

    assert(neighborPt_ai >= 0 && neighborPt_bi >= 0);
    assert(partTri_a->neighbors[neighborPt_ai]->triangle == ta);
    assert(partTri_b->neighbors[neighborPt_bi]->triangle == tb);

    partTri_a->neighbors[neighborPt_ai] = partTri_b;
    partTri_b->neighbors[neighborPt_bi] = partTri_a;

    ta->partialTriangles[ca_i].neighbors[neighbor_ai] = &tb->partialTriangles[cb_i];
    tb->partialTriangles[cb_i].neighbors[neighbor_bi] = &ta->partialTriangles[ca_i];

}
