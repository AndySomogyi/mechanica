/*
 * MeshOperations.cpp
 *
 *  Created on: Nov 20, 2017
 *      Author: andy
 */

#include "MeshOperations.h"
#include "MxMesh.h"
#include <algorithm>
#include <limits>
#include <iostream>


RadialEdgeSplit::RadialEdgeSplit(MeshPtr mesh, float _longCutoff, const Edge& _edge) :
    MeshOperation{mesh}, longCutoff{_longCutoff}, edge{_edge} {

    float range = mesh->edgeSplitStochasticAsymmetry / 2;
    uniformDist = std::uniform_real_distribution<float>(-range, range);
}

bool RadialEdgeSplit::applicable(const Edge& e) {
    return true;
}

RadialEdgeCollapse::RadialEdgeCollapse(MeshPtr mesh, float _shortCutoff, const Edge& _edge) :
    MeshOperation{mesh}, shortCutoff{_shortCutoff}, edge{_edge}
{
}

bool RadialEdgeCollapse::applicable(const Edge& _e) {
    MxEdge edge{_e};
        // check if we have a manifold edge, most common kind of short edge
    if (edge.upperFacets().size() == 0 &&
        edge.lowerFacets().size() == 0 &&
        edge.radialFacets().size() == 1) {
        return true;
    }

    std::cout << "only manifold edge collapse supported" << std::endl;
    return false;
}

EdgeFlip::EdgeFlip(MeshPtr mesh, const Edge& _edge) : MeshOperation{mesh} {
}

bool EdgeFlip::applicable(const Edge& _e) {
    return false;
}


/**
 * The std *_heap functions produce a max heap. We need the lowest energy items
 * first, so flip the compare op here.
 */
struct MeshOperationComp
{
    bool operator()(const MeshOperation* a, const MeshOperation *b) const {
        return a->energy() > b->energy();
    };
};

MeshOperations::MeshOperations(MeshPtr _mesh, float _shortCutoff,
        float _longCutoff) :
    mesh{_mesh}, shortCutoff{_shortCutoff}, longCutoff{_longCutoff}
{
}

HRESULT MeshOperations::positionsChanged(TriangleContainer::const_iterator triBegin,
        TriangleContainer::const_iterator triEnd) {

    for(auto iter = triBegin; iter != triEnd; ++iter) {
        const TrianglePtr tri = *iter;
        int minIndx = -1, maxIndx = -1;
        float minEdge = std::numeric_limits<float>::max();
        float maxEdge = std::numeric_limits<float>::min();
        for(int i = 0; i < 3; ++i) {
            float dist = Magnum::Math::distance(tri->vertices[i]->position, tri->vertices[(i+1)%3]->position);
            if(dist < minEdge) {
                minEdge = dist;
                minIndx = i;
            }
            if(dist > maxEdge) {
                maxEdge = dist;
                maxIndx = i;
            }
        }

        Edge edge = {{tri->vertices[minIndx], tri->vertices[(minIndx+1)%3]}};

        if(minEdge < shortCutoff &&
                findMatchingOperation(edge) == nullptr &&
                RadialEdgeCollapse::applicable(edge)) {
            push(new RadialEdgeCollapse(mesh, shortCutoff, edge));
        }

        else {
            edge = {{tri->vertices[maxIndx], tri->vertices[(maxIndx+1)%3]}};

            if(maxEdge > longCutoff &&
                    findMatchingOperation(edge) == nullptr &&
                    RadialEdgeSplit::applicable(edge)) {
                push(new RadialEdgeSplit(mesh, longCutoff, edge));
            }
        }
    }

    return S_OK;
}

HRESULT MeshOperations::valenceChanged(const VertexPtr vert) {
    return E_NOTIMPL;
}

HRESULT MeshOperations::removeDependentOperations(const TrianglePtr tri) {
    Container::iterator iter = c.begin();
    while((iter = findDependentOperation(iter, tri)) != c.end()) {
        MeshOperation *op = *iter;
        delete op;
        c.erase(iter++);
    }
    // stuff was removed, need to re-order the heap.
    std::make_heap(c.begin(), c.end(), MeshOperationComp{});
    return S_OK;
}

HRESULT MeshOperations::removeDependentOperations(const VertexPtr vert) {
    Container::iterator iter = c.begin();
    while((iter = findDependentOperation(iter, vert)) != c.end()) {
        MeshOperation *op = *iter;
        iter = c.erase(iter);
        delete op;
    }
    // stuff was removed, need to re-order the heap.
    std::make_heap(c.begin(), c.end(), MeshOperationComp{});
    return S_OK;
}

HRESULT MeshOperations::apply() {
    MeshOperation *op;
    HRESULT res = S_OK;

    while((op = pop()) != nullptr) {
        if(res == S_OK) {
            res = op->apply();
        }
        delete op;
    }

    return res;
}


void MeshOperations::push(MeshOperation* x) {
    c.push_back(x);
    std::push_heap(c.begin(), c.end(), MeshOperationComp{});
}

MeshOperation* MeshOperations::pop() {
    if(c.size() > 0) {
        // moves the largest to the back
        std::pop_heap(c.begin(), c.end(), MeshOperationComp{});
        MeshOperation *result = c.back();
        c.pop_back();
        return result;
    }
    return nullptr;
}

void MeshOperations::setShortCutoff(float val) {
    shortCutoff = val;
}

void MeshOperations::setLongCutoff(float val) {
    longCutoff = val;
}

MeshOperations::Container::iterator MeshOperations::findDependentOperation(
        MeshOperations::Container::iterator start, const TrianglePtr tri) {
    return std::find_if(start, c.end(),
            [tri](const MeshOperation* op)->bool { return op->depends(tri); });
}

MeshOperations::Container::iterator MeshOperations::findDependentOperation(
        MeshOperations::Container::iterator start, const VertexPtr vert) {
    return std::find_if(start, c.end(),
            [vert](const MeshOperation* op)->bool { return op->depends(vert); });
}

MeshOperation* MeshOperations::findMatchingOperation(const Edge& edge) {
    Container::iterator iter = std::find_if(c.begin(), c.end(),
            [edge](const MeshOperation* op)->bool { return op->equals(edge); });
    return iter != c.end() ? *iter : nullptr;
}



static int ctr = 0;

/**
 * go around the ring of the edge, and split every incident triangle on
 * that edge. Creates a new vertex at the midpoint of this edge.
 */

HRESULT RadialEdgeSplit::apply() {
    MxEdge e{edge};

    auto triangles = e.radialTriangles();

    assert(triangles.size() >= 2);

#ifndef NDEBUG
    std::vector<TrianglePtr> newTriangles;
#endif

    ctr += 1;

    // new vertex at the center of this edge
    Vector3 center = (e.a->position + e.b->position) / 2.;
    center = center + (e.a->position - e.b->position) * uniformDist(randEngine);
    VertexPtr vert = mesh->createVertex(center);

    TrianglePtr firstNewTri = nullptr;
    TrianglePtr prevNewTri = nullptr;

    for(uint i = 0; i < triangles.size(); ++i)
    {
        TrianglePtr tri = triangles[i];
        std::cout << "tri[" << i << "], cell[0]:" << tri->cells[0] << ", cell[1]:" << tri->cells[1] << std::endl;
    }

    for(uint i = 0; i < triangles.size(); ++i)
    {
        TrianglePtr tri = triangles[i];

        #ifndef NDEBUG
        float originalArea = tri->area;
        #endif

        // find the outside tri vertex
        VertexPtr outer = nullptr;
        for(uint i = 0; i < 3; ++i) {
            if(tri->vertices[i] !=  e.b && tri->vertices[i] != e.a ) {
                outer = tri->vertices[i];
                break;
            }
        }
        assert(outer);

        // copy of old triangle vertices, replace the bottom (a) vertex
        // here with the new center vertex
        auto vertices = tri->vertices;
        for(uint i = 0; i < 3; ++i) {
            if(vertices[i] == e.a) {
                vertices[i] = vert;
                break;
            }
        }

        // the original triangle has three vertices in the order
        // {a, outer, b}, in CCW winding order. The new vertices in the same
        // winding order are {vert, outer, b}. Because we ensure the same winding
        // we can set up the cell pointers and partial triangles in the same
        // order.
        TrianglePtr nt = mesh->createTriangle((MxTriangleType*)tri->ob_type, vertices);
        nt->cells = tri->cells;
        nt->partialTriangles[0].ob_type = tri->partialTriangles[0].ob_type;
        nt->partialTriangles[1].ob_type = tri->partialTriangles[1].ob_type;

#ifndef NDEBUG
        newTriangles.push_back(nt);
#endif

        // make damned sure the winding is correct and the new triangle points
        // in the same direction as the existing one
        assert(Math::dot(nt->normal, tri->normal) >= 0);


        if(tri->facet) {
            tri->facet->appendChild(nt);
        }

        // remove the b vertex from the old triangle, and replace it with the
        // new center vertex
        disconnect(tri, e.b);
        connect(tri, vert);

        tri->positionsChanged();

        // make damned sure the winding is correct and the new triangle points
        // in the same direction as the existing one, but now with the
        // replaced b vertex
        assert(Math::dot(nt->normal, tri->normal) >= 0);

        // make sure at most 1% difference in new total area and original area.
        assert(std::abs(nt->area + tri->area - originalArea) < (1.0 / originalArea));

        // makes sure that new and old tri share an edge.
        assert(adjacent(tri, nt));

        // removes the e.b - outer edge connection connection from the old
        // triangle and replaces it with the new triangle,
        // manually add the partial triangles to the cell
        for(uint i = 0; i < 2; ++i) {
            if(tri->cells[i] != mesh->rootCell()) {

                assert(tri->partialTriangles[i].unboundNeighborCount() == 0);
                assert(nt->partialTriangles[i].unboundNeighborCount() == 3);
                reconnect(&tri->partialTriangles[i], &nt->partialTriangles[i], {{e.b, outer}});
                assert(tri->partialTriangles[i].unboundNeighborCount() == 1);
                assert(nt->partialTriangles[i].unboundNeighborCount() == 2);
                connect(&tri->partialTriangles[i], &nt->partialTriangles[i]);
                assert(tri->partialTriangles[i].unboundNeighborCount() == 0);
                assert(nt->partialTriangles[i].unboundNeighborCount() == 1);
                tri->cells[i]->boundary.push_back(&nt->partialTriangles[i]);
                if(tri->cells[i]->renderer) {
                    tri->cells[i]->renderer->invalidate();
                }
            }
        }

        assert(incident(nt, {{e.b, outer}}));
        assert(!incident(tri, {{e.b, outer}}));


        // split the mass according to area
        nt->mass = nt->area / (nt->area + tri->area) * tri->mass;
        tri->mass = tri->area / (nt->area + tri->area) * tri->mass;

        if(i == 0) {
            firstNewTri = nt;
            prevNewTri = nt;
        } else {
            #ifndef NDEBUG
            if (ctr >= 109 && triangles.size() >= 4) {
                std::cout << "boom" << std::endl;
            }
            #endif
            connect(nt, prevNewTri);
            prevNewTri = nt;
        }
    }

    // connect the first and last new triangles. If this is a
    // manifold edge, only 2 new triangles, which already got
    // connected above.
    if(triangles.size() > 2) {
        if(triangles.size() >= 4) {
            std::cout << "root cell: " << mesh->rootCell() << std::endl;
            std::cout << "boom" << std::endl;
        }
        connect(firstNewTri, prevNewTri);
    }



#ifndef NDEBUG
    for(uint t = 0; t < newTriangles.size(); ++t) {
        TrianglePtr nt = newTriangles[t];
        assert(nt->isValid());
        for(uint i = 0; i < 2; ++i) {
            if(nt->cells[i] != mesh->rootCell()) {
                assert(nt->partialTriangles[i].unboundNeighborCount() == 0);
            }
        }
    }
    for(uint t = 0; t < triangles.size(); ++t) {
        TrianglePtr tri = triangles[t];
        for(uint i = 0; i < 2; ++i) {
            if(tri->cells[i] != mesh->rootCell()) {
                assert(tri->partialTriangles[i].unboundNeighborCount() == 0);
            }
        }
    }
    for(uint t = 0; t < triangles.size(); ++t) {
        TrianglePtr tri = triangles[t];
        for(uint i = 0; i < 2; ++i) {
            if(tri->cells[i] != mesh->rootCell()) {
                assert(tri->cells[i]->manifold());
            }
        }
    }

    mesh->validate();
#endif

    return S_OK;
}

float RadialEdgeSplit::energy() const {
    return -(Magnum::Math::distance(edge[0]->position, edge[1]->position) - longCutoff);
}

bool RadialEdgeSplit::depends(const TrianglePtr tri) const {
    for(int i = 0; i < 3; ++i) {
        if(equals({{tri->vertices[i], tri->vertices[(i+1)%3]}})) {
            return true;
        }
    }
    return false;
}

bool RadialEdgeSplit::depends(const VertexPtr v) const {
    return v == edge[0] || v == edge[1];
}

bool RadialEdgeSplit::equals(const Edge& e) const {
    return (e[0] == edge[0] && e[1] == edge[1]) ||
           (e[0] == edge[1] && e[1] == edge[0]);
}


static void testTriangle(const TrianglePtr tri) {
    assert(isfinite(tri->area) && tri->area > 0);
    assert(isfinite(tri->aspectRatio) && tri->aspectRatio > 0);
    assert(isfinite(tri->mass) && tri->mass > 0);
    assert(isfinite(tri->normal.length()) && tri->normal.length() > 0.9 && tri->normal.length() < 1.1);
    assert(tri->cells[0] && tri->cells[1]);
}

static int collapseStr = 0;


/**
 * is this configuration topologically safe to reconnect. This check that the top
 * and bottom triangle neighbors are not themselves connected.
 */
static HRESULT safeTopology(const TrianglePtr tri, const Edge& edge1, const Edge& edge2) {
    for(int i = 0; i < 2; ++i) {
        if(!tri->cells[i]->isRoot()) {

            PTrianglePtr p1 = nullptr, p2 = nullptr;

            for(int j = 0; j < 3; ++j) {
                PTrianglePtr pn = tri->partialTriangles[i].neighbors[j];
                if(incident(pn, edge1)) {
                    p1 = pn;
                    continue;
                }
                if(incident(pn, edge2)) {
                    p2 = pn;
                    continue;
                }
            }

            assert(p1 && p2);
            assert(p1 != p2);
            assert(p1 != &tri->partialTriangles[i]);
            assert(p2 != &tri->partialTriangles[i]);
            assert(adjacent(p1, &tri->partialTriangles[i]));
            assert(adjacent(p2, &tri->partialTriangles[i]));

            if (adjacent(p1, p2)) {
                return mx_error(E_FAIL, "can't perform edge collapse, not topologically invariant");
            }
        }
    }
    return S_OK;
};

/**
 * Remove the partial triangles pointers from the given triangle,
 * and it's two neighboring partial on the given two edges,
 * and reconnect the two outer neighboring partial triangles with each other.
 * do this for both sides of the triangle.
 */
static void reconnectPartialTriangles(TrianglePtr tri, const Edge& edge1, const Edge& edge2) {
    for(int i = 0; i < 2; ++i) {
        if(!tri->cells[i]->isRoot()) {

            PTrianglePtr p1 = nullptr, p2 = nullptr;

            assert(incident(tri, edge1));
            assert(incident(tri, edge2));
            assert(incident(&tri->partialTriangles[i], edge1));
            assert(incident(&tri->partialTriangles[i], edge2));

            for(int j = 0; j < 3; ++j) {
                assert(tri->partialTriangles[i].neighbors[j]);

                PTrianglePtr pn = tri->partialTriangles[i].neighbors[j];
                if(incident(pn, edge1)) {
                    p1 = pn;
                    continue;
                }
                if(incident(pn, edge2)) {
                    p2 = pn;
                    continue;
                }
            }

            assert(p1 && p2);
            assert(p1 != p2);
            assert(p1 != &tri->partialTriangles[i]);
            assert(p2 != &tri->partialTriangles[i]);
            assert(adjacent(p1, &tri->partialTriangles[i]));
            assert(adjacent(p2, &tri->partialTriangles[i]));

            assert(!adjacent(p1, p2));

            disconnect(p1, &tri->partialTriangles[i]);
            disconnect(p2, &tri->partialTriangles[i]);
            connect(p1, p2);
            assert(tri->partialTriangles[i].unboundNeighborCount() == 2);
        }
    }
};

static void moveMaterial(const EdgeTriangles& leftTri,
        const EdgeTriangles& rightTri,
        float leftFraction, float rightFraction,
        const TrianglePtr src)  {
    assert(abs(1.0 - (leftFraction + rightFraction)) < 0.01);
    int leftCount = 0;
    int rightCount = 0;

    for(auto i = leftTri.begin(); i != leftTri.end(); ++i) {
        leftCount += 1;
    }

    for(auto i = rightTri.begin(); i != rightTri.end(); ++i) {
        rightCount += 1;
    }

    assert(leftCount > 0 && rightCount > 0);

    for(TrianglePtr tri : leftTri) {
        tri->mass += (leftFraction * src->mass) / leftCount;
    }

    for(TrianglePtr tri : rightTri) {
        tri->mass += (rightFraction * src->mass) / rightCount;
    }
};

static HRESULT collapseTriangleOnEdge(MeshPtr mesh, TrianglePtr t1, const Edge& edge) {
    HRESULT res;

    collapseStr++;

    MxEdge e{edge};

    // the opposite vertex from the collapse edge
    VertexPtr c = nullptr;

#ifndef NDEBUG
    auto cells = t1->cells;
    if(!cells[0]->isRoot()) assert(cells[0]->manifold());
    if(!cells[1]->isRoot()) assert(cells[1]->manifold());
#endif

    for(VertexPtr v : t1->vertices) {
        if(v != e.a && v != e.b) {
            c = v;
        }
        if(v->facets().size() != 1) {
        //    return mx_error(E_FAIL, "vertex belongs to more than one facet");
        }
    }



    assert(c);

    auto t3 = EdgeTriangles(t1, t1->adjacentEdgeIndex(e.a, c));
    auto t4 = EdgeTriangles(t1, t1->adjacentEdgeIndex(e.b, c));


    // new center position
    Magnum::Vector3 pos = (e.a->position + e.b->position) / 2;

    // all of the triangles attached to edge endpoints a and b will have their corner
    // that is attached to the edge endpoints moved to the new center. Need to
    // check all of these triangles and make sure that we do not invert any triangles,
    // or cause any triangles to become colinear (zero area).


    // is it safe to move this triangle
    auto safeTriangleMove = [pos, t1](const TrianglePtr tri, const VertexPtr vert) -> HRESULT {

        if(tri == t1) {
            return S_OK;
        }

        Vector3 before = normal(tri->vertices[0]->position,
                tri->vertices[1]->position,
                tri->vertices[2]->position);

        Vector3 pos0 = (tri->vertices[0] == vert) ? pos : tri->vertices[0]->position;
        Vector3 pos1 = (tri->vertices[1] == vert) ? pos : tri->vertices[1]->position;
        Vector3 pos2 = (tri->vertices[2] == vert) ? pos : tri->vertices[2]->position;

        Vector3 after = normal(pos0, pos1, pos2);

        if(Magnum::Math::dot(before, after) <= 0) {
            return mx_error(E_FAIL, "can't perform edge collapse, triangle will become inverted");
        }

        if(Magnum::Math::triangle_area(pos0, pos1, pos2) < 0.001) {
            return mx_error(E_FAIL, "can't perform edge collapse, triangle area becomes too small or colinear");
        }

        return S_OK;
    };

    for(TrianglePtr tri : e.a->triangles()) {
        if((res = safeTriangleMove(tri, e.a)) != S_OK) {
            return res;
        }
    }

    for(TrianglePtr tri : e.b->triangles()) {
        if((res = safeTriangleMove(tri, e.b)) != S_OK) {
            return res;
        }
    }

    // make sure with topologically safe
    if((res = safeTopology(t1, {{e.a, c}}, {{e.b, c}})) != S_OK) return res;

    float leftUpperArea = Magnum::Math::triangle_area(e.a->position, c->position, pos);
    float rightUpperArea = Magnum::Math::triangle_area(e.b->position, c->position, pos);

    // need to calculate area here, because the area in the triangle has not been updated yet.
    float upperArea = Magnum::Math::triangle_area(e.a->position, e.b->position, c->position);


    assert(leftUpperArea > 0 && rightUpperArea > 0);



    moveMaterial(t3, t4, leftUpperArea/upperArea, rightUpperArea/upperArea, t1);




#ifndef NDEBUG
    if(!cells[0]->isRoot()) assert(cells[0]->manifold());
    if(!cells[1]->isRoot()) assert(cells[1]->manifold());

    for(TrianglePtr tri : e.a->triangles()) {
        assert(tri->cells[0] && tri->cells[1]);
    }
    for(TrianglePtr tri : e.b->triangles()) {
        assert(tri->cells[0] && tri->cells[1]);
    }
#endif

    // disconnect the partial triangle pointers.
    reconnectPartialTriangles(t1, {{e.a, c}}, {{e.b, c}});


    VertexPtr vsrc = nullptr, vdest = nullptr;

    if(e.a->triangles().size() >= e.b->triangles().size()) {
        vsrc = e.b;
        vdest = e.a;
    } else {
        vsrc = e.a;
        vdest = e.b;
    }

    // the destination vertex where all the other triangles get moved to,
    // set this to the new center pos.
    vdest->position = pos;

    vdest->removeTriangle(t1);
    vsrc->removeTriangle(t1);


    std::vector<TrianglePtr> srcTriangles = vsrc->triangles();

    for(int i = 0; i < vdest->triangles().size(); ++i) {
        TrianglePtr tri = vdest->triangles()[i];
        testTriangle(tri);
    }

    for(int i = 0; i < vsrc->triangles().size(); ++i) {
        TrianglePtr tri = vsrc->triangles()[i];
        testTriangle(tri);
    }

    for(TrianglePtr tri : srcTriangles) {
        if(tri == t1) continue;

        testTriangle(tri);

        disconnect(tri, vsrc);
        connect(tri, vdest);

        assert(tri->cells[0] && tri->cells[1]);

        tri->positionsChanged();

        testTriangle(tri);

        for(int i = 0; i < 2; ++i) {
            tri->cells[i]->topologyChanged();
        }
    }

    disconnect(t1, e.a);
    disconnect(t1, e.b);
    disconnect(t1, c);

    assert(t1->vertices[0] == nullptr && t1->vertices[1] == nullptr && t1->vertices[2] == nullptr);


    mesh->deleteVertex(vsrc);

#ifndef NDEBUG
    for(TrianglePtr tri : mesh->triangles) {
        if(tri == t1) continue;
        assert(!incident(tri, vsrc));
    }

    for(int i = 0; i < vdest->triangles().size(); ++i) {
        TrianglePtr tri = vdest->triangles()[i];
        assert(tri != t1);
        assert(tri->isValid());
    }
#endif

    for(int i = 0; i < 2; ++i) {
        CellPtr cell = t1->cells[i];
        cell->removeChild(t1);
        cell->topologyChanged();
        cell->topologyChanged();
    }

    mesh->deleteTriangle(t1);


#ifndef NDEBUG
    if(!cells[0]->isRoot()) assert(cells[0]->manifold());
    if(!cells[1]->isRoot()) assert(cells[1]->manifold());
    mesh->validate();
#endif

    return S_OK;

}


HRESULT RadialEdgeCollapse::apply() {

    HRESULT res;

    collapseStr++;

    TrianglePtr t1 = nullptr, t2 = nullptr;

    MxEdge e{edge};

    std::vector<TrianglePtr> edgeTri = e.radialTriangles();

    assert(edgeTri.size() == 2);

    t1 = edgeTri[0]; t2 = edgeTri[1];

    VertexPtr c = nullptr, d = nullptr;

#ifndef NDEBUG
    auto cells = t1->cells;
    assert(cells == t2->cells);

    if(!cells[0]->isRoot()) assert(cells[0]->manifold());
    if(!cells[1]->isRoot()) assert(cells[1]->manifold());
#endif

    for(VertexPtr v : t1->vertices) {
        if(v != e.a && v != e.b) {
            c = v;
        }
        if(v->facets().size() != 1) {
        //    return mx_error(E_FAIL, "vertex belongs to more than one facet");
        }
    }

    for(VertexPtr v : t2->vertices) {
        if(v != e.a && v != e.b) {
            d = v;
        }
        if(v->facets().size() != 1) {
        //    return mx_error(E_FAIL, "vertex belongs to more than one facet");
        }
    }

    assert(c && d);

    auto t3 = EdgeTriangles(t1, t1->adjacentEdgeIndex(e.a, c));
    auto t4 = EdgeTriangles(t1, t1->adjacentEdgeIndex(e.b, c));
    auto t5 = EdgeTriangles(t2, t2->adjacentEdgeIndex(e.a, d));
    auto t6 = EdgeTriangles(t2, t2->adjacentEdgeIndex(e.b, d));

    // new center position
    Magnum::Vector3 pos = (e.a->position + e.b->position) / 2;

    // all of the triangles attached to edge endpoints a and b will have their corner
    // that is attached to the edge endpoints moved to the new center. Need to
    // check all of these triangles and make sure that we do not invert any triangles,
    // or cause any triangles to become colinear (zero area).


    // is it safe to move this triangle
    auto safeTriangleMove = [pos, t1, t2](const TrianglePtr tri, const VertexPtr vert) -> HRESULT {

        if((tri == t1) || (tri == t2)) {
            return S_OK;
        }

        Vector3 before = normal(tri->vertices[0]->position,
                tri->vertices[1]->position,
                tri->vertices[2]->position);

        Vector3 pos0 = (tri->vertices[0] == vert) ? pos : tri->vertices[0]->position;
        Vector3 pos1 = (tri->vertices[1] == vert) ? pos : tri->vertices[1]->position;
        Vector3 pos2 = (tri->vertices[2] == vert) ? pos : tri->vertices[2]->position;

        Vector3 after = normal(pos0, pos1, pos2);

        if(Magnum::Math::dot(before, after) <= 0) {
            return mx_error(E_FAIL, "can't perform edge collapse, triangle will become inverted");
        }

        if(Magnum::Math::triangle_area(pos0, pos1, pos2) < 0.001) {
            return mx_error(E_FAIL, "can't perform edge collapse, triangle area becomes too small or colinear");
        }

        return S_OK;
    };

    for(TrianglePtr tri : e.a->triangles()) {
        if((res = safeTriangleMove(tri, e.a)) != S_OK) {
            return res;
        }
    }

    for(TrianglePtr tri : e.b->triangles()) {
        if((res = safeTriangleMove(tri, e.b)) != S_OK) {
            return res;
        }
    }

    // make sure with topologically safe
    if((res = safeTopology(t1, {{e.a, c}}, {{e.b, c}})) != S_OK) return res;
    if((res = safeTopology(t2, {{e.a, d}}, {{e.b, d}})) != S_OK) return res;


    float leftUpperArea = Magnum::Math::triangle_area(e.a->position, c->position, pos);
    float leftLowerArea = Magnum::Math::triangle_area(e.a->position, d->position, pos);
    float rightUpperArea = Magnum::Math::triangle_area(e.b->position, c->position, pos);
    float rightLowerArea = Magnum::Math::triangle_area(e.b->position, d->position, pos);

    // need to calculate area here, because the area in the triangle has not been updated yet.
    float upperArea = Magnum::Math::triangle_area(e.a->position, e.b->position, c->position);
    float lowerArea = Magnum::Math::triangle_area(e.a->position, e.b->position, d->position);

    assert(leftLowerArea > 0 && leftUpperArea > 0 && rightLowerArea > 0 && rightUpperArea > 0);


    moveMaterial(t3, t4, leftUpperArea/upperArea, rightUpperArea/upperArea, t1);
    moveMaterial(t5, t6, leftLowerArea/lowerArea, rightLowerArea/lowerArea, t2);



#ifndef NDEBUG
    if(!cells[0]->isRoot()) assert(cells[0]->manifold());
    if(!cells[1]->isRoot()) assert(cells[1]->manifold());

    for(TrianglePtr tri : e.a->triangles()) {
        assert(tri->cells[0] && tri->cells[1]);
    }
    for(TrianglePtr tri : e.b->triangles()) {
        assert(tri->cells[0] && tri->cells[1]);
    }
#endif

    // disconnect the partial triangle pointers.
    reconnectPartialTriangles(t1, {{e.a, c}}, {{e.b, c}});
    reconnectPartialTriangles(t2, {{e.a, d}}, {{e.b, d}});

    VertexPtr vsrc = nullptr, vdest = nullptr;

    if(e.a->triangles().size() >= e.b->triangles().size()) {
        vsrc = e.b;
        vdest = e.a;
    } else {
        vsrc = e.a;
        vdest = e.b;
    }

    // the destination vertex where all the other triangles get moved to,
    // set this to the new center pos.
    vdest->position = pos;

    vdest->removeTriangle(t1);
    vdest->removeTriangle(t2);
    vsrc->removeTriangle(t1);
    vsrc->removeTriangle(t2);

    std::vector<TrianglePtr> srcTriangles = vsrc->triangles();

    for(int i = 0; i < vdest->triangles().size(); ++i) {
        TrianglePtr tri = vdest->triangles()[i];
        testTriangle(tri);
    }

    for(int i = 0; i < vsrc->triangles().size(); ++i) {
        TrianglePtr tri = vsrc->triangles()[i];
        testTriangle(tri);
    }

    for(TrianglePtr tri : srcTriangles) {
        if(tri == t1 || tri == t2) continue;

        testTriangle(tri);

        disconnect(tri, vsrc);
        connect(tri, vdest);

        assert(tri->cells[0] && tri->cells[1]);

        tri->positionsChanged();

        testTriangle(tri);

        for(int i = 0; i < 2; ++i) {
            tri->cells[i]->topologyChanged();
        }
    }

    disconnect(t1, e.a);
    disconnect(t1, e.b);
    disconnect(t2, e.a);
    disconnect(t2, e.b);
    disconnect(t1, c);
    disconnect(t2, d);

    assert(t1->vertices[0] == nullptr && t1->vertices[1] == nullptr && t1->vertices[2] == nullptr);
    assert(t2->vertices[0] == nullptr && t2->vertices[1] == nullptr && t2->vertices[2] == nullptr);

    mesh->deleteVertex(vsrc);

#ifndef NDEBUG
    for(TrianglePtr tri : mesh->triangles) {
        if(tri == t1 || tri == t2) continue;
        assert(!incident(tri, vsrc));
    }

    for(int i = 0; i < vdest->triangles().size(); ++i) {
        TrianglePtr tri = vdest->triangles()[i];
        assert(tri != t1 && tri != t2);
        assert(tri->isValid());
    }
#endif

    for(int i = 0; i < 2; ++i) {
        CellPtr cell = t1->cells[i];
        cell->removeChild(t1);
        cell->topologyChanged();
        cell = t2->cells[i];
        cell->removeChild(t2);
        cell->topologyChanged();
    }

    mesh->deleteTriangle(t1);
    mesh->deleteTriangle(t2);

#ifndef NDEBUG
    if(!cells[0]->isRoot()) assert(cells[0]->manifold());
    if(!cells[1]->isRoot()) assert(cells[1]->manifold());
    mesh->validate();
#endif

    return S_OK;
}

float RadialEdgeCollapse::energy() const {
    return (1 - shortCutoff / Magnum::Math::distance(edge[0]->position, edge[1]->position));
}

bool RadialEdgeCollapse::depends(const TrianglePtr tri) const {
    for(int i = 0; i < 3; ++i) {
        if(equals({{tri->vertices[i], tri->vertices[(i+1)%3]}})) {
            return true;
        }
    }
    return false;
}

bool RadialEdgeCollapse::depends(const VertexPtr v) const {
    return v == edge[0] || v == edge[1];
}

bool RadialEdgeCollapse::equals(const Edge& e) const {
    return (e[0] == edge[0] && e[1] == edge[1]) ||
           (e[0] == edge[1] && e[1] == edge[0]);
}

MeshOperations::~MeshOperations() {
}
