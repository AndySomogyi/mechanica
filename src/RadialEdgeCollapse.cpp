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

static bool haveDebugObj = false;
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
            if(!haveDebugObj) return;
            std::cout << "radial tri 0: ";
            debugTriangle(gtri[0]);
            break;
        case '1':
            if(!haveDebugObj) return;
            std::cout << "radial tri 1: ";
            debugTriangle(gtri[1]);
            break;
        case '2':
            if(!haveDebugObj) return;
            std::cout << "radial tri 2: ";
            debugTriangle(gtri[2]);
            break;
        case 'a':
            if(!haveDebugObj) return;
            mesh->makeTrianglesTransparent();
            gtri[0].tri->color = Magnum::Color4{1, 1, 0, 1};
            break;
        case 'b':
            if(!haveDebugObj) return;
            mesh->makeTrianglesTransparent();
            gtri[1].tri->color = Magnum::Color4{1, 1, 0, 1};
            break;
        case 'c':
            if(!haveDebugObj) return;
            mesh->makeTrianglesTransparent();
            gtri[2].tri->color = Magnum::Color4{1, 1, 0, 1};
            break;
        case 'e':
            if(!haveDebugObj) return;
            mesh->makeTrianglesTransparent();
            gtri[0].tri->color = Magnum::Color4{1, 1, 0, 1};
            gtri[1].tri->color = Magnum::Color4{1, 1, 0, 1};
            gtri[2].tri->color = Magnum::Color4{1, 1, 0, 1};
            break;
        case 'n':
            for(TrianglePtr tri : mesh->triangles) {
                tri->color[3] = 0;
                tri->alpha = 0.3;
            }
            break;
        case 'd':
            ops->setDebugMode(true);
            break;
        case 'o':
            ops->setDebugMode(false);
            break;
        case 's':
            ops->debugStep();
            break;
    }
    std::cout << "char: " << (char)c << std::endl;
}



RadialEdgeCollapse::RadialEdgeCollapse(MeshPtr mesh, float _shortCutoff, const Edge& _edge) :
    MeshOperation{mesh}, shortCutoff{_shortCutoff}, edge{_edge}
{
}

MeshOperation *RadialEdgeCollapse::create(MeshPtr mesh, float shortCutoff, const Edge& _e) {

    return new RadialEdgeCollapse(mesh, shortCutoff, _e);

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
    assert(isfinite(tri->getMass()) && tri->getMass() > 0);
    assert(isfinite(tri->normal.length()) && tri->normal.length() > 0.9 && tri->normal.length() < 1.1);

    for(int i = 0; i < 2; ++i) {
        if(tri->cells[i]->isRoot()) {
            assert(tri->partialTriangles[i].mass == 0.);
        } else {
            isfinite(tri->partialTriangles[i].mass ) && tri->partialTriangles[i].mass  > 0;
        }
    }
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

            if (adjacent(p1, p2)) {
                return mx_error(E_FAIL, "can't perform edge collapse, not topologically invariant");
            }
        }
    }
    return S_OK;
};

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

        // need to check EVERY vertex of the triangle, as a triangle can be attached
        // to the edge endpoint via one vertex, but it still has two remaining
        // vertices that can connect with triangles from the other edge
        // endpoint.
        for(int i = 0; i < 3; ++i) {
            if(tri->vertices[i] != edge[0] && tri->vertices[i] != edge[1]) {
                if(edgeLink.find(tri->vertices[i]) == edgeLink.end()) {
                    leftLink.insert(tri->vertices[i]);
                }
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
 *     center: the partial triangle that's being removed.
 *     edge: the edge that is being collapsed, edge[0] is the
 *           left vertex, edge[1] is the right.
 *     leftFrac: fraction of triangle area on the left side of split
 *     rightTrac: same thing
 *     apex: the apex vertex
 */
static void reconnectPartialTriangles(PTrianglePtr center,
        VertexPtr vsrc, VertexPtr vdest, float leftFrac,
        float rightFrac, VertexPtr apex ) {

    // first find the two remaining partial triangles, the left and right ones.
    PTrianglePtr pLeft = nullptr, pRight = nullptr;

    assert(incident(center, {{vsrc, vdest}}));
    assert(incident(center, {{vsrc, apex}}));
    assert(incident(center, {{vdest, apex}}));

    int l = center->triangle->adjacentEdgeIndex(apex, vsrc);
    int r = center->triangle->adjacentEdgeIndex(apex, vdest);

    assert(l >= 0 && r >= 0);

    pLeft = center->neighbors[l];
    pRight = center->neighbors[r];

    assert(pLeft && pRight);
    assert(incident(pLeft->triangle, {{apex, vsrc}}));
    assert(incident(pRight->triangle, {{apex, vdest}}));
    assert(!adjacent(pLeft, pRight));

    disconnect_partial_triangles(center, pLeft);
    disconnect_partial_triangles(center, pRight);

    assert(center->unboundNeighborCount() >= 2);
    assert(pLeft->unboundNeighborCount() == 1);
    assert(pRight->unboundNeighborCount() == 1);

    assert(isfinite(leftFrac) && leftFrac > 0);
    assert(isfinite(rightFrac) && rightFrac > 0);
    //assert(isfinite(center->mass) && center->mass > 0);

    pLeft->mass += leftFrac * center->mass;
    pRight->mass += rightFrac * center->mass;

    for(int i = 0; i < 3; ++i) {
        if(pLeft->neighbors[i] == nullptr) {
            pLeft->neighbors[i] = pRight;
        }
        if(pRight->neighbors[i] == nullptr) {
            pRight->neighbors[i] = pLeft;
        }
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
        const VertexPtr vsrc, const VertexPtr vdest, const Vector3 &pos) {

    // Move the material from this triangle to it's four connected
    // surface partial triangles that belong to the upper and lower
    // cell surfaces
    float leftArea = Magnum::Math::triangle_area(vsrc->position,
            rt.apex->position, pos);
    float rightArea = Magnum::Math::triangle_area(vdest->position,
            rt.apex->position, pos);

    // need to calculate area here, because the area in the triangle
    // has not been updated yet.
    float totalArea = Magnum::Math::triangle_area(vsrc->position,
            vdest->position, rt.apex->position);

    assert(leftArea > 0 && rightArea > 0);

    // Reconnect the partial triangles. This function performs the
    // partial triangle material moving.
    for(int i = 0; i < 2; ++i) {
        reconnectPartialTriangles(&rt.tri->partialTriangles[i],
                vsrc, vdest, leftArea / totalArea, rightArea / totalArea,
                rt.apex);
    }

    disconnect_triangle_vertex(rt.tri, vsrc);
    disconnect_triangle_vertex(rt.tri, vdest);
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



HRESULT RadialEdgeCollapse::apply() {
    HRESULT res;

    ctr++;

    std::cout << "collapsing radial edge {" << edge[0]->id << ", " << edge[1]->id << "}" << std::endl;

#ifdef NOISY
    std::cout << "Edge[0]:" << std::endl;
    for(TrianglePtr tri : edge[0]->triangles()) {
        assert(tri->isValid());
        std::cout << tri << std::endl;
    }

    std::cout << "Edge[1]:" << std::endl;
    for(TrianglePtr tri : edge[1]->triangles()) {
        assert(tri->isValid());
        std::cout << tri << std::endl;
    }
#endif

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

#ifdef NOISY
    std::cout << "edge triangles:" << std::endl;
    for(auto tri : edgeTri) {
        std::cout << tri << std::endl;
    }
    std::cout << "doing it..." << std::endl;
#endif

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
        res = collapseTriangleOnEdge(mesh, triangles[i], vsrc, vdest, pos);
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

        // side effect of removing triangle from vertex tri list.
        disconnect_triangle_vertex(tri, vsrc);

        // adds tri to vert tri list.
        connect_triangle_vertex(tri, vdest);

        assert(tri->cells[0] && tri->cells[1]);

        tri->positionsChanged();
    }

#ifndef NDEBUG
    bool badTri = false;
    for(TrianglePtr tri : vdest->triangles()) {
        if(!tri->isValid()) {
            std::cout << "bad triangle" << tri << std::endl;
            badTri = true;
        }
    }
    assert(!badTri);
#endif



    // done with the src vertex
    assert(vsrc->triangles().size() == 0);
    mesh->deleteVertex(vsrc);

    // notify the attached cells that the topology has changed
    for(CellPtr cell : cells)
    {
        assert(cell->isValid());
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


void RadialEdgeCollapse::mark() const {

    std::cout << "marking radial edge collapse edge {" << edge[0]->id << ", " << edge[1]->id << "}" << std::endl;

    for(TrianglePtr tri : mesh->triangles) {
        tri->color[3] = 0;
        tri->alpha = 0.3;
    }

    EdgeTriangles triangles(edge);

    for(TrianglePtr tri : triangles) {
        tri->color = Magnum::Color4::yellow();
    }
}
