/*
 * RadialEdgeCollapse.cpp
 *
 *  Created on: Nov 23, 2017
 *      Author: andy
 */

#include "RadialEdgeCollapse.h"
#include "MxMesh.h"
#include <iostream>


RadialEdgeCollapse::RadialEdgeCollapse(MeshPtr mesh, float _shortCutoff, const Edge& _edge) :
    MeshOperation{mesh}, shortCutoff{_shortCutoff}, edge{_edge}
{
}

bool RadialEdgeCollapse::applicable(const Edge& _e) {

    return true;

    /*
    MxEdge edge{_e};
        // check if we have a manifold edge, most common kind of short edge
    if (edge.upperFacets().size() == 0 &&
        edge.lowerFacets().size() == 0 &&
        edge.radialFacets().size() == 1) {
        return true;
    }

    std::cout << "only manifold edge collapse supported" << std::endl;
    return false;
    */
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
 * Check all of the incident triangles to this vertex, and check if moving
 * the vertex inverts any triangles.
 *
 * Ignore the radial edge triangles, as these will be deleted, and
 * they could be very thin, triggering the second colinear condition.
 */
static HRESULT canVertexBeMoved(const VertexPtr v, const Vector3& pos, const Triangles& ignore) {
    for(const TrianglePtr tri : v->triangles()) {

        if(contains(ignore, tri)) {
            return S_OK;
        }

        Vector3 before = normal(tri->vertices[0]->position,
                tri->vertices[1]->position,
                tri->vertices[2]->position);

        Vector3 pos0 = (tri->vertices[0] == v) ? pos : tri->vertices[0]->position;
        Vector3 pos1 = (tri->vertices[1] == v) ? pos : tri->vertices[1]->position;
        Vector3 pos2 = (tri->vertices[2] == v) ? pos : tri->vertices[2]->position;

        Vector3 after = normal(pos0, pos1, pos2);

        if(Magnum::Math::dot(before, after) <= 0) {
            return mx_error(E_FAIL, "can't perform edge collapse, triangle will become inverted");
        }

        if(Magnum::Math::triangle_area(pos0, pos1, pos2) < 0.001) {
            return mx_error(E_FAIL, "can't perform edge collapse, triangle area becomes too small or colinear");
        }
    }

    return S_OK;
}

static HRESULT canTriangleBeCollapsed(const TrianglePtr, const Edge& e) {

}

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

            disconnect_partial_triangles(p1, &tri->partialTriangles[i]);
            disconnect_partial_triangles(p2, &tri->partialTriangles[i]);
            connect_partial_triangles(p1, p2);
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


/**
 * Collapses the edge of a single triangle, and reconnect all of the triangles
 * on the two remaining edges to each other.
 */

static HRESULT collapseTriangleOnEdge(MeshPtr mesh, TrianglePtr t1, const Edge& edge) {
    HRESULT res;

    collapseStr++;

    MxEdge e{edge};

    int test[collapseStr];

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

        disconnect_triangle_vertex(tri, vsrc);
        connect_triangle_vertex(tri, vdest);

        assert(tri->cells[0] && tri->cells[1]);

        tri->positionsChanged();

        testTriangle(tri);

        for(int i = 0; i < 2; ++i) {
            tri->cells[i]->topologyChanged();
        }
    }

    disconnect_triangle_vertex(t1, e.a);
    disconnect_triangle_vertex(t1, e.b);
    disconnect_triangle_vertex(t1, c);

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


/**
 * RadialEdgeCollapse Section
 */




HRESULT RadialEdgeCollapse::apply() {

    HRESULT res;

    collapseStr++;

    TrianglePtr t1 = nullptr, t2 = nullptr;

    MxEdge e{edge};

    // check if we have a manifold edge, most common kind of short edge
    if (e.upperFacets().size() != 0 ||
        e.lowerFacets().size() != 0 ||
        e.radialFacets().size() != 1) {
        std::cout << "waring, only manifold edge collapse currently supported." << std::endl;
        return E_FAIL;
    }

    int test[e.radialTriangles().size()];

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

        disconnect_triangle_vertex(tri, vsrc);
        connect_triangle_vertex(tri, vdest);

        assert(tri->cells[0] && tri->cells[1]);

        tri->positionsChanged();

        testTriangle(tri);

        for(int i = 0; i < 2; ++i) {
            tri->cells[i]->topologyChanged();
        }
    }

    disconnect_triangle_vertex(t1, e.a);
    disconnect_triangle_vertex(t1, e.b);
    disconnect_triangle_vertex(t2, e.a);
    disconnect_triangle_vertex(t2, e.b);
    disconnect_triangle_vertex(t1, c);
    disconnect_triangle_vertex(t2, d);

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
