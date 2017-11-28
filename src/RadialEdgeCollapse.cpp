/*
 * RadialEdgeCollapse.cpp
 *
 *  Created on: Nov 23, 2017
 *      Author: andy
 */

#include "RadialEdgeCollapse.h"
#include "MxMesh.h"
#include "MxDebug.h"
#include <iostream>
#include <set>

#ifdef _WIN32
#include <malloc.h>
#define alloca(SIZE)  _alloca(SIZE)
#define
#else
#include <alloca.h>
#endif

MeshOperations *ops = nullptr;
MeshPtr mesh = nullptr;


static Edge e;

static int ctr = 0;


static void markEdge(const Edge& edge) {
    EdgeTriangles triangles(edge);

    for(TrianglePtr tri : triangles) {
        tri->color = Magnum::Color4::green();
    }
}



/**
 * It's expensive to interrogate each triangle, accesses a lot of info that
 * is used both the check if the configuration is valid, and to reconnect
 * it. Save in the info here on the query pass, and reuse it on the
 * reconnect pass.
 */
struct RadialTriangle {
    TrianglePtr tri = nullptr;
    VertexPtr apex = nullptr;
    std::vector<TrianglePtr> leftTriangles;
    std::vector<TrianglePtr> rightTriangles;

};

std::vector<RadialTriangle> gtri;

static void debugTriangle(RadialTriangle &rt) {

    mesh->makeTrianglesTransparent();

    rt.tri->color = Magnum::Color4{1, 1, 0, 1};

    std::cout << "id{" << rt.tri->id << "}, cells:{" << rt.tri->cells[0]->id << ", " << rt.tri->cells[1]->id << "}," << std::endl <<
       "\tpos{" <<
        rt.tri->vertices[0]->position << ", " <<
        rt.tri->vertices[1]->position << ", " <<
        rt.tri->vertices[2]->position << "}" << std::endl;

    int i = 0;
    for(TrianglePtr tri : rt.leftTriangles) {
        tri->color = Magnum::Color4::green();
        std::cout << "left tri :" << i << ", id{" << tri->id << "}, cells:{" << tri->cells[0]->id << ", " << tri->cells[1]->id << "}," << std::endl
        << "\tpos{" <<
        tri->vertices[0]->position << ", " <<
        tri->vertices[1]->position << ", " <<
        tri->vertices[2]->position << "}" << std::endl;
        ++i;
    }

    i = 0;
    for(TrianglePtr tri : rt.rightTriangles) {
        tri->color = Magnum::Color4::red();
        std::cout << "right tri:" << i << ", id{" << tri->id << "}, cells:{" << tri->cells[0]->id << ", " << tri->cells[1]->id << "}," << std::endl
        << "\tpos{" <<
        tri->vertices[0]->position << ", " <<
        tri->vertices[1]->position << ", " <<
        tri->vertices[2]->position << "}" << std::endl;
        ++i;
    }

}

void setMeshOpDebugMode(uint c) {

    switch (c) {
        case '0':
            std::cout << "radial tri 0: ";
            debugTriangle(gtri[0]);
            break;
        case '1':
            std::cout << "radial tri 1: ";
            debugTriangle(gtri[1]);
            break;
        case '2':
            std::cout << "radial tri 2: ";
            debugTriangle(gtri[2]);
            break;
        case 'a':
            mesh->makeTrianglesTransparent();
            gtri[0].tri->color = Magnum::Color4{1, 1, 0, 1};
            break;
        case 'b':
            mesh->makeTrianglesTransparent();
            gtri[1].tri->color = Magnum::Color4{1, 1, 0, 1};
            break;
        case 'c':
            mesh->makeTrianglesTransparent();
            gtri[2].tri->color = Magnum::Color4{1, 1, 0, 1};
            break;
        default:
            break;
    }
    std::cout << "char: " << (char)c << std::endl;
}



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


/**
 * is this configuration topologically safe to reconnect. This check that the top
 * and bottom triangle neighbors are not themselves connected.
 */
static HRESULT safeTopology(const TrianglePtr tri, const Edge& edge1, const Edge& edge2) {

    assert(tri->isValid());

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

            //if(ctr == 81) {
                //p1->triangle->color = Magnum::Color4{1, 0, 0, 0.6};
                //p2->triangle->color = Magnum::Color4{1, 0, 0, 0.6};
            //}

            bool ptAdj = adjacent(p1, p2);
            bool triAdj = adjacent(p1->triangle, p2->triangle);

            if (adjacent(p1, p2)) {
                return mx_error(E_FAIL, "can't perform edge collapse, not topologically invariant");
            }
        }
    }
    return S_OK;
};

static HRESULT checkTrapezoid(const std::vector<RadialTriangle> &rt) {

    std::set<const TrianglePtr> tris;

    for(const RadialTriangle& t : rt) {
        for(const TrianglePtr tri : t.leftTriangles) {
            if (tris.find(tri) == tris.end()) {
                tris.insert(tri);
            } else {
                return mx_error(E_FAIL, "trapezoid edge collapse not supported yet");
            }
        }

        for(const TrianglePtr tri : t.rightTriangles) {
            if (tris.find(tri) == tris.end()) {
                tris.insert(tri);
            } else {
                return mx_error(E_FAIL, "trapezoid edge collapse not supported yet");
            }
        }

    }



    return S_OK;
}


/**
 * Determine if this radial edge meets the link condition.
 *
 * In a triangular mesh, the *star* of a vertex v is the is the set of triangles
 * and edges that are incident to v. The link of a vertex is the frontier of the
 * star.
 */
static HRESULT radialLinkCondition(const Edge& edge, const EdgeTriangles& triangles) {

    // when we look at the link of the edge vertices, if the triangle incident
    // to the edge is a radial triangle, we don't bother looking at it's
    // vertices because each vertex is either already in the edge link, or
    // is one of the edge vertices.
    std::set<TrianglePtr> radialTriangles;

    std::set<VertexPtr> edgeLink;

    for(TrianglePtr tri : triangles) {
        radialTriangles.insert(tri);

        // find the apex vertex, the edge link is the set of all apex verticies
        for(int i = 0; i < 3; ++i) {
            if(tri->vertices[i] != edge[0] && tri->vertices[i] != edge[1]) {
                edgeLink.insert(tri->vertices[i]);
                break;
            }
        }
    }

    // any vertex contained within the link of each vertex will already
    // be in the link of the edge, so we don't bother inserting them.
    std::set<VertexPtr> leftLink;

    for(TrianglePtr tri : edge[0]->triangles()) {

        // if the triangle is a radial triangle, ignore it.
        if(radialTriangles.find(tri) != radialTriangles.end()) {
            continue;
        }

        for(int i = 0; i < 3; ++i) {
            if(tri->vertices[i] != edge[0] && tri->vertices[i] != edge[1]) {
                if(edgeLink.find(tri->vertices[i]) == edgeLink.end()) {
                    leftLink.insert(tri->vertices[i]);
                }
                break;
            }
        }
    }

    for(TrianglePtr tri : edge[1]->triangles()) {
        if(radialTriangles.find(tri) != radialTriangles.end()) {
            continue;
        }

        for(int i = 0; i < 3; ++i) {
            if(tri->vertices[i] != edge[0] && tri->vertices[i] != edge[1]) {
                if(leftLink.find(tri->vertices[i]) != leftLink.end() &&
                   edgeLink.find(tri->vertices[i]) == edgeLink.end()) {
                    markEdge(edge);
                    return mx_error(E_FAIL, "edge triangle violates condition, can't perform edge collapse");
                }
            }
        }
    }

    return S_OK;
}



/**
 * Check all of the incident triangles to this vertex, and check if moving
 * the vertex inverts any triangles.
 *
 * Ignore the radial edge triangles, as these will be deleted, and
 * they could be very thin, triggering the second colinear condition.
 */
static HRESULT canTriangleVertexBeMoved(const TrianglePtr tri,
        const VertexPtr v, const Vector3 &pos) {

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

    return S_OK;
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

/**
 * Remove the partial triangles pointers from the given center
 * partial triangle. Reconnects the remaining partial triangles
 * to each other.
 *
 * Takes the material on the center partial triangle, and moves it to the
 * two neighboring partial triangles that remain.
 *
 * Works even in the case where we are collapsing a tetrahedron. The
 * material gets moved to the two neighboring partial triangles, and
 * then, in a later step, these partial triangles get deleted,
 * and their material gets moved down to neighbors that remain.
 *
 * @param
 *     center: the partial triangle to keep
 *     edge: the edge that is being collapsed, edge[0] is the
 *           left vertex, edge[1] is the right.
 *     leftFrac: fraction of triangle area on the left side of split
 *     rightTrac: same thing
 *     apex: the apex vertex
 */
static void reconnectPartialTriangles(PTrianglePtr center,
        const Edge& edge, float leftFrac,
        float rightFrac, VertexPtr apex ) {

    // first find the two remaining partial triangles, the left and right ones.
    PTrianglePtr pLeft = nullptr, pRight = nullptr;

    assert(incident(center, edge));
    assert(incident(center, {{edge[0], apex}}));
    assert(incident(center, {{edge[1], apex}}));

    for(int j = 0; j < 3; ++j) {
        PTrianglePtr pn = center->neighbors[j];
        if(!pn) {
            continue;
        }

        if(incident(pn, {{edge[0], apex}})) {
            pLeft = pn;
            continue;
        }
        if(incident(pn, {{edge[1], apex}})) {
            pRight = pn;
            continue;
        }
    }

    assert(pLeft && pRight);

    assert(!adjacent(pLeft, pRight));

    disconnect_partial_triangles(center, pLeft);
    disconnect_partial_triangles(center, pRight);
    connect_partial_triangles(pLeft, pRight);
    assert(center->unboundNeighborCount() >= 2);

    pLeft->mass += leftFrac * center->mass;
    pRight->mass += rightFrac * center->mass;
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
 *
 * Note, this function does not move the connected triangles, that's done
 * later for purposes of efficiency.
 *
 *
 * tasks:
 *     * move the material from the two facing partial triangles, and
 *       reapportion the material to the four connected partial triangles.
 *
 *     * disconnect the four PTs from the center tri, and reconnect the
 *       PTs to each other.
 *
 *     * remove the tri from all of its vertices
 *
 *     * delete the triangle.
 *
 * preconditions:
 *     * the edge vertex to be kept already has it's position changed
 *       to the new center position.
 */
static HRESULT collapseTriangleOnEdge(MeshPtr mesh, RadialTriangle &rt,
        const VertexPtr vsrc, const VertexPtr vdest, const Edge& edge,
        const Vector3 &pos) {

    // Move the material from this triangle to it's four connected
    // surface partial triangles that belong to the upper and lower
    // cell surfaces
    float leftArea = Magnum::Math::triangle_area(edge[0]->position,
            rt.apex->position, pos);
    float rightArea = Magnum::Math::triangle_area(edge[1]->position,
            rt.apex->position, pos);

    // need to calculate area here, because the area in the triangle
    // has not been updated yet.
    float totalArea = Magnum::Math::triangle_area(edge[1]->position,
            edge[1]->position, rt.apex->position);

    assert(leftArea > 0 && rightArea > 0);

    // Reconnect the partial triangles. This function performs the
    // partial triangle material moving.
    for(int i = 0; i < 2; ++i) {
        if(!rt.tri->cells[i]->isRoot()) {
            reconnectPartialTriangles(&rt.tri->partialTriangles[i],
                    edge, leftArea / totalArea, rightArea / totalArea,
                    rt.apex);
        }
    }

    disconnect_triangle_vertex(rt.tri, edge[0]);
    disconnect_triangle_vertex(rt.tri, edge[1]);
    disconnect_triangle_vertex(rt.tri, rt.apex);

    assert(rt.tri->vertices[0] == nullptr &&
           rt.tri->vertices[1] == nullptr &&
           rt.tri->vertices[2] == nullptr);

    for(int i = 0; i < 2; ++i) {
        CellPtr cell = rt.tri->cells[i];
        cell->removeChild(rt.tri);
    }

    // delete the triangle here, this will not affect the previously
    // cached triangle edges for manifold like triangles.
    mesh->deleteTriangle(rt.tri);

    return S_OK;
}

/**
 * Determine if the triangle can be collapsed or not. If it can, save the triangle
 * info in the RadialTriangle ptr.
 *
 * returns S_OK if the triangle can be collapsed, an error otherwise.
 */
static HRESULT classifyRadialTriangle(TrianglePtr tri,
        const Edge& edge, const Vector3 &pos, RadialTriangle &res) {


    res.tri = tri;

    for(int i = 0; i < 3; ++i) {
        if(tri->vertices[i] != edge[0] && tri->vertices[i] != edge[1]) {
            res.apex = tri->vertices[i];
        }
    }

    assert(res.apex);
    assert(incident(tri,edge));
    assert(incident(tri, {{edge[0], res.apex}}));
    assert(incident(tri, {{edge[1], res.apex}}));


    if(ctr == 81) {
        std::cout << "foo" << std::endl;
    }

    assert(res.apex);

    EdgeTriangles leftTriangles{{{edge[0], res.apex}}};
    EdgeTriangles rightTriangles{{{edge[1], res.apex}}};

    assert(leftTriangles.size() >= 2);
    assert(rightTriangles.size() >= 2);
    res.leftTriangles.resize(leftTriangles.size() - 1);
    res.rightTriangles.resize(rightTriangles.size() - 1);

    {
        int triIndx = 0;
        for(TrianglePtr t : leftTriangles) {
            if (t != tri) {
                res.leftTriangles[triIndx++] = t;
            }
        }

        triIndx = 0;
        for(TrianglePtr t : rightTriangles) {
            if (t != tri) {
                res.rightTriangles[triIndx++] = t;
            }
        }
    }
    
    /*

    std::cout << "leftTriSize: " << res.leftTriangles.size() << std::endl;
    for(int i = 0; i < res.leftTriangles.size(); ++i) {
        for(int j = 0; j < 2; ++j) {
            for(int k = 0; k < 2; ++k) {
                std::cout << "leftTri[" << i << "].ptri[" << j << "] adj to tri.ptri[" << k << "]: " <<
                adjacent(&res.leftTriangles[i]->partialTriangles[j], &tri->partialTriangles[k]) << std::endl;
            }
        }
    }

    std::cout << "rightTriSize: " << res.rightTriangles.size() << std::endl;
    for(int i = 0; i < res.rightTriangles.size(); ++i) {
        for(int j = 0; j < 2; ++j) {
            for(int k = 0; k < 2; ++k) {
                std::cout << "rightTri[" << i << "].ptri[" << j << "] adj to tri.ptri[" << k << "]: " <<
                adjacent(&res.rightTriangles[i]->partialTriangles[j], &tri->partialTriangles[k]) << std::endl;
            }
        }
    }
     
     */


    // check if a geometry move would invert any adjacent triangles
    for(TrianglePtr t : res.leftTriangles) {
        if(t != tri) {
            HRESULT r = canTriangleVertexBeMoved(t, edge[0], pos);
            if(r != S_OK) return r;
        }
    }

    for(TrianglePtr t : res.rightTriangles) {
        if(t != tri) {
            HRESULT r = canTriangleVertexBeMoved(t, edge[1], pos);
            if(r != S_OK) return r;
        }
    }

    return safeTopology(tri, {{edge[0], res.apex}}, {{edge[1], res.apex}});
}


HRESULT RadialEdgeCollapse::oldApply() {

    HRESULT res;

    TrianglePtr t1 = nullptr, t2 = nullptr;

    MxEdge e{edge};

    // check if we have a manifold edge, most common kind of short edge
    if (e.upperFacets().size() != 0 ||
        e.lowerFacets().size() != 0 ||
        e.radialFacets().size() != 1) {
        std::cout << "waring, only manifold edge collapse currently supported." << std::endl;
        return E_FAIL;
    }

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
    }

    for(VertexPtr v : t2->vertices) {
        if(v != e.a && v != e.b) {
            d = v;
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

        // side effect of removing triangle from vertex tri list.
        disconnect_triangle_vertex(tri, vsrc);

        // adds tri to vert tri list.
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



HRESULT RadialEdgeCollapse::newApply() {
    HRESULT res;

    ctr++;

    if(ctr == 81) {
        std::cout << "foo" << std::endl;
        ops->stop(edge);
        e = edge;
    }



    ::mesh = this->mesh;
    ops = &mesh->meshOperations;

    // all of the cells that are incident to this edge, will need to notify
    // all of them that the topology has changed.
    std::set<CellPtr> cells;

    // new center position
    Magnum::Vector3 pos = (edge[0]->position + edge[1]->position) / 2;

    // iterate over all of the non-radial edge triangles, and check if they
    // can be moved.
    for(const TrianglePtr tri : edge[0]->triangles()) {
        if(!incident(tri, edge[1]) &&
                (res = canTriangleVertexBeMoved(tri, edge[0], pos)) != S_OK) {
            return res;
        }
        assert(tri->cells[0] && tri->cells[1]);
        cells.insert(tri->cells[0]);
        cells.insert(tri->cells[1]);
    }

    for(const TrianglePtr tri : edge[1]->triangles()) {
        if(!incident(tri, edge[0]) &&
                (res = canTriangleVertexBeMoved(tri, edge[1], pos)) != S_OK) {
            return res;
        }
        assert(tri->cells[0] && tri->cells[1]);
        cells.insert(tri->cells[0]);
        cells.insert(tri->cells[1]);
    }

    // at this point, all of the incident triangle moves will be geometrically valid,
    // now check if we can topologically perform this operation

    EdgeTriangles edgeTri{edge};

    if((res = radialLinkCondition(edge, edgeTri)) != S_OK) {
        return res;
    }

    const uint edgeTriSize = edgeTri.size();

    //RadialTriangle *triangles = (RadialTriangle*)alloca(
    //        edgeTriSize * sizeof(RadialTriangle));
    std::vector<RadialTriangle> triangles(edgeTriSize);

    {
        uint i = 0;
        for(auto iter : edgeTri) {
            assert(i < edgeTriSize);
            if((res = classifyRadialTriangle(iter, edge, pos, triangles[i]))
                    != S_OK) {
                return res;
            }
            i += 1;
        }
    }

    if((res = checkTrapezoid(triangles)) != S_OK) {
        return res;
    }

    //for(int i = 0; i < edgeTriSize; ++i) {
    //    for(TrianglePtr tri : triangles[i].leftTriangles) {

    //    }
   // }

/*

    if(ctr == 81) {

        gtri = triangles;


        triangles[0].tri->color = Magnum::Color4{0, 0, 0, 1};
        triangles[1].tri->color = Magnum::Color4{1, 1, 0, 1};
        triangles[2].tri->color = Magnum::Color4{0, 1, 0, 1};

        for(int i = 0; i < 3; ++i) {
            std::cout << "tri:" << i << ", cells:{"
            << triangles[i].tri->cells[0]->id << ", " << triangles[i].tri->cells[1]->id << "}" << std::endl;
        }


        mesh->makeTrianglesTransparent();

        for(TrianglePtr tri : triangles[2].leftTriangles) {
            tri->color = Magnum::Color4::green();
        }

        for(TrianglePtr tri : triangles[2].rightTriangles) {
            tri->color = Magnum::Color4::red();
        }

        return E_FAIL;
    }
 */


    // source and destination vertices, where we detach and attach one side of edge to.
    VertexPtr vsrc = nullptr, vdest = nullptr;
    if(edge[0]->triangles().size() >= edge[1]->triangles().size()) {
        vsrc = edge[1];
        vdest = edge[0];
    } else {
        vsrc = edge[0];
        vdest = edge[1];
    }

    // collapse the radial edge triangle. This also removes the
    // triangle from both the edge vertices, and deletes the triangle.
    for(uint i = 0; i < edgeTriSize; ++i) {
        res = collapseTriangleOnEdge(mesh, triangles[i], vsrc, vdest, edge, pos);
        assert(res == S_OK);
    }

    // move all of the triangles that were attached to the src to the
    // destination. There can be more triangles attached to the src
    // than just the radial edge triangles, these need to be moved
    // also.

    // the destination vertex where all the other triangles get moved to,
    // set this to the new center pos.
    vdest->position = pos;

    std::vector<TrianglePtr> srcTriangles = vsrc->triangles();

#ifndef NDEBUG
    for(int i = 0; i < vdest->triangles().size(); ++i) {
        TrianglePtr tri = vdest->triangles()[i];
        testTriangle(tri);
    }

    for(int i = 0; i < vsrc->triangles().size(); ++i) {
        TrianglePtr tri = vsrc->triangles()[i];
        testTriangle(tri);
    }
#endif

    for(TrianglePtr tri : srcTriangles) {

        testTriangle(tri);

        // side effect of removing triangle from vertex tri list.
        disconnect_triangle_vertex(tri, vsrc);

        // adds tri to vert tri list.
        connect_triangle_vertex(tri, vdest);

        assert(tri->cells[0] && tri->cells[1]);

        tri->positionsChanged();

        testTriangle(tri);
    }

    // done with the src vertex
    assert(vsrc->triangles().size() == 0);
    mesh->deleteVertex(vsrc);

    // notify the attached cells that the topology has changed
    for(CellPtr cell : cells)
    {
        assert(cell->manifold());
        cell->topologyChanged();
    }

#ifndef NDEBUG
    std::vector<TrianglePtr> dtri = vdest->triangles();
    for(int i = 0; i < dtri.size(); ++i) {
        TrianglePtr tri = dtri[i];
        testTriangle(tri);
        //assert(tri->validate());

        for(int i = 0; i < 3; ++i) {
            EdgeTriangles et{{{tri->vertices[i], tri->vertices[(i+1)%3]}}};
            if(!et.isValid()) {
                ops->stop({{tri->vertices[i], tri->vertices[(i+1)%3]}});
            }
        }
    }

    mesh->validateTriangles();
#endif

    return S_OK;
}


HRESULT RadialEdgeCollapse::apply() {
    return newApply();
}

