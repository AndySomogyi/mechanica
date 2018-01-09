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

struct ValenceCell {
    CellPtr cell;
    int valenceCnt;
};

static ValenceCell lowestValenceCell(CVertexPtr vertex);





/**
 * A vertex with three or more cell valences is either a skeletal vertex, or a
 * cone vertex.
 */


/**
 * Splits a vertex in the direction of the cell.
 */
static HRESULT splitVertex(MeshPtr, VertexPtr, CCellPtr);

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


MeshOperation *VertexSplit::create(MeshPtr mesh, VertexPtr v)
{
    //if (vertexSplitId > 1) return nullptr;

    
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
    
    //verifyVertex(vertex);
#endif
    
    int size = vertex->cells().size();
    /*
    if(id > 0) {
        for(TrianglePtr tri : vertex->triangles()) {
            if(tri->cells[0]->id == 4 || tri->cells[1]->id == 4) {
                tri->color = Color4{1.0, 0., 0., 1.0};
            }
        }
        return S_OK;
    }
     */
    
#ifndef NDEBUG
    cnt++;
    std::vector<CellPtr> cells = vertex->cells();
#endif

    if(size <= 3 || (size <= 3 && !sharedCellCount(vertex))) {
        return S_OK;
    }
    

        ValenceCell lowest = lowestValenceCell(vertex);
        
        if(lowest.valenceCnt + 1 >= size) {
            std::cout << "can't split vertex, don't have a low valence count cell" << std::endl;
            return S_OK;
        }
    
    std::cout << "VertexSplit::apply(), vertex: " << vertex << std::endl
    << "cell count: " << size << ", " << std::endl
    << "sharedCellCount: " << sharedCellCount(vertex) << std::endl
    << "nonRootCellCount: " << nonRootCellCount(vertex) << std::endl
    << "lowestValenceCount: " << lowest.valenceCnt << std::endl
    << "lowestValenceCell: " << lowest.cell->id << std::endl;
    
    if(lowest.valenceCnt <= 1) {
        std::cout << "hmmm..." << std::endl;
    }

    // find the cell with the max force
    //CCellPtr cell = maxForceCell(vertex);
    
    HRESULT result = splitVertex(mesh, vertex, lowest.cell);
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
static HRESULT splitVertex(MeshPtr mesh, VertexPtr center, CCellPtr cell) {

    // fan by definition must have at least three triangles
    std::vector<TrianglePtr> fan = triangleFan(center, cell);
    
    std::vector<TrianglePtr> newTris;
    
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
    Vector3 diff = fanCentroid - center->position;
    
    float distance = mesh->getShortCutoff() >= 0.001 ? 2 * mesh->getShortCutoff() : 0.002;
    VertexPtr newVertex = mesh->createVertex(center->position + distance * diff.normalized());

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
            
            newTris.push_back(tri);
        }
    }

    for(TrianglePtr tri : fan) {
        replaceTriangleVertex(tri, center, newVertex);
    }
    
    if(newTris.size()) {

        assert(firstNewTri && tri && firstTriConnectCell);
        connect_triangle_partial_triangles(firstNewTri, tri, firstTriConnectCell);
    
        for(TrianglePtr tri : newTris) {
            tri->positionsChanged();
            assert(tri->isValid());
        }
    }

#ifndef NDEBUG
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
    


#endif
    
    std::cout << "finished splitting vertex, center vertex: " << std::endl
    << center << std::endl
    << ", new vertex: " << newVertex << std::endl
    << ", new vertex cell count: " << newVertex->cells().size() << std::endl
    << ", new vertex shared cell count: " << sharedCellCount(newVertex) << std::endl
    << ", center vertex: " << center << std::endl
    << ", center vertex cell count: " << center->cells().size() << std::endl
    << ", center vertex shared cell count: " << sharedCellCount(center) << std::endl;

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
        if(!c->isRoot()) {
            cells.insert(c);
        }
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



CCellPtr maxForceCell(CVertexPtr v)
{
    static int cnt = 0;
    cnt++;
    
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
            if(gaussCurv > maxCurv) {
                maxCurv = meanCurv;
                cell = c;
            }
        }

    }
    
    std::cout << "vertex split pull cell: " << cell->id << std::endl;
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

    TrianglePtr newTriangle = mesh->createTriangle({{c0, c1}},
            {{newVertex, centerVertex, frontierVertex}});
    
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
    
    newTriangle->color = Color4{1., 0., 0., 1.};
    


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
    
    /*
    std::cout << "VertexSplit created new triangle: " << std::endl;
    std::cout << newTriangle << std::endl;
     */
    
    return newTriangle;
}

static ValenceCell lowestValenceCell(CVertexPtr vertex) {
    ValenceCell result = {nullptr, std::numeric_limits<int>::max()};

    for(CellPtr cell : vertex->cells()) {

        CellSet cells;

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
            cells.insert(otherCell);

            TrianglePtr next = tri->nextTriangleInFan(vertex, cell, prev);
            prev = tri;
            tri = next;
        } while(tri && tri != first);

        if(cells.size() < result.valenceCnt) {
            result.valenceCnt = cells.size();
            result.cell = cell;
        }
    }
    assert(result.cell);
    return result;
}
