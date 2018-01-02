/*
 * RadialEdgeSplit.cpp
 *
 *  Created on: Nov 23, 2017
 *      Author: andy
 */

#include "RadialEdgeSplit.h"
#include "MxMesh.h"
#include <iostream>


RadialEdgeSplit::RadialEdgeSplit(MeshPtr mesh, float _longCutoff, const Edge& _edge) :
    MeshOperation{mesh}, longCutoff{_longCutoff}, edge{_edge} {

    float range = mesh->edgeSplitStochasticAsymmetry / 2;
    uniformDist = std::uniform_real_distribution<float>(0,0);
}

bool RadialEdgeSplit::applicable(const Edge& e) {
    return true;
}




static int ctr = 0;

/**
 * go around the ring of the edge, and split every incident triangle on
 * that edge. Creates a new vertex at the midpoint of this edge.
 */

HRESULT RadialEdgeSplit::apply() {
    std::cout << "splitting radial edge {" << edge[0]->id << ", " << edge[1]->id << "} {" << std::endl;

    EdgeTriangles e{edge};

    std::vector<TrianglePtr> triangles(e.begin(), e.end());

    /**
     * Find the common cell between a pair of triangles. If there are only
     * two trianges, the 0'th common tri will be the cell that's on the
     * 0 side of the 0'th triangle, and the common cell will be on the
     * 1 side of the 1st triangle.
     */
    auto commonCell = [this, &triangles] (int indx) -> CCellPtr {
        if(triangles.size() == 2) {
            assert(indx == 0 || indx == 1);
            return triangles[0]->cells[indx];
        }
        
        // wrap negative index around to end of array
        const int size = triangles.size();
        int prevIndx = ((indx-1)%size+size)%size;
        
        CTrianglePtr tri = triangles[indx];
        CTrianglePtr prevTri = triangles[prevIndx];
        
        for(int i = 0; i < 2; ++i) {
            for(int j = 0; j < 3; ++j) {
                if(tri->partialTriangles[i].neighbors[j]->triangle == prevTri) {
                    return tri->cells[i];
                }
            }
        }
        assert(0 && "something really bad happened");
        return nullptr;
    };

    assert(triangles.size() >= 2);

#ifndef NDEBUG
    std::vector<TrianglePtr> newTriangles;
#endif

    ctr += 1;
    std::cout << "ctr:" << ctr << std::endl;
    if(ctr >= 616) {
        std::cout << "p" << std::endl;
        int t = 0;
        for(TrianglePtr tri : e) {
            std::cout << "triangle[" << t++ << "] to split: " << tri << std::endl;
        }
    }

    // new vertex at the center of this edge
    Vector3 center = (edge[0]->position + edge[1]->position) / 2.;
    center = center + (edge[1]->position - edge[1]->position) * uniformDist(randEngine);
    VertexPtr vert = mesh->createVertex(center);

    TrianglePtr firstNewTri = nullptr;
    TrianglePtr prevNewTri = nullptr;

#ifdef NOISY
    for(uint i = 0; i < triangles.size(); ++i)
    {
        TrianglePtr tri = triangles[i];
        std::cout << tri << std::endl;
    }
#endif
    std::cout << "}" << std::endl;

    for(uint i = 0; i < triangles.size(); ++i)
    {
        TrianglePtr tri = triangles[i];

        #ifndef NDEBUG
        float originalArea = tri->area;
        #endif

        // find the outside tri vertex
        VertexPtr outer = nullptr;
        for(uint i = 0; i < 3; ++i) {
            if(tri->vertices[i] !=  edge[1] && tri->vertices[i] != edge[0] ) {
                outer = tri->vertices[i];
                break;
            }
        }
        assert(outer);

        // copy of old triangle vertices, replace the bottom (a) vertex
        // here with the new center vertex
        auto vertices = tri->vertices;
        for(uint i = 0; i < 3; ++i) {
            if(vertices[i] == edge[0]) {
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

        // remove the b vertex from the old triangle, and replace it with the
        // new center vertex
        disconnect_triangle_vertex(tri, edge[1]);
        connect_triangle_vertex(tri, vert);

        tri->positionsChanged();

        // make damned sure the winding is correct and the new triangle points
        // in the same direction as the existing one, but now with the
        // replaced b vertex
        assert(Math::dot(nt->normal, tri->normal) >= 0);

        // make sure at most 1% difference in new total area and original area.
        assert(std::abs(nt->area + tri->area - originalArea) < (1.0 / originalArea));

        // makes sure that new and old tri share an edge.
        assert(adjacent_triangle_vertices(tri, nt));

        // removes the edge[1] - outer edge connection connection from the old
        // triangle and replaces it with the new triangle,
        // manually add the partial triangles to the cell
        for(uint i = 0; i < 2; ++i) {
            assert(tri->partialTriangles[i].unboundNeighborCount() == 0);
            assert(nt->partialTriangles[i].unboundNeighborCount() == 3);
            reconnect(&tri->partialTriangles[i], &nt->partialTriangles[i], {{edge[1], outer}});
            assert(tri->partialTriangles[i].unboundNeighborCount() == 1);
            assert(nt->partialTriangles[i].unboundNeighborCount() == 2);
            connect_partial_triangles(&tri->partialTriangles[i], &nt->partialTriangles[i]);
            assert(tri->partialTriangles[i].unboundNeighborCount() == 0);
            assert(nt->partialTriangles[i].unboundNeighborCount() == 1);
            tri->cells[i]->boundary.push_back(&nt->partialTriangles[i]);
            if(tri->cells[i]->renderer) {
                tri->cells[i]->renderer->invalidate();
            }
        }

        assert(incident(nt, {{edge[1], outer}}));
        assert(!incident(tri, {{edge[1], outer}}));


        // split the mass according to area
        for(int i = 0; i < 2; ++i) {
            nt->partialTriangles[i].mass = nt->area / (nt->area + tri->area) * tri->partialTriangles[i].mass;
            tri->partialTriangles[i].mass = tri->area / (nt->area + tri->area) * tri->partialTriangles[i].mass;
        }

        assert(isfinite(nt->getMass()) && nt->getMass() > 0);
        assert(isfinite(tri->getMass()) && tri->getMass() > 0);

        if(i == 0) {
            firstNewTri = nt;
            prevNewTri = nt;
        } else {
            assert(adjacent_triangle_vertices(nt, prevNewTri));
            connect_triangle_partial_triangles(nt, prevNewTri, commonCell(i));
            prevNewTri = nt;
        }
    }

    // connect the first and last new triangles. If this edge has two triangles,
    // one side of the triangles was connected in the above loop, and this connects
    // the other side.
    connect_triangle_partial_triangles(firstNewTri, prevNewTri, commonCell(0));
    
#ifndef NDEBUG
    for(uint t = 0; t < newTriangles.size(); ++t) {
        TrianglePtr nt = newTriangles[t];
        std::cout << nt;
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

    for(TrianglePtr tri : edge[0]->triangles()) {
        assert(tri->isValid());
    }

    for(TrianglePtr tri : edge[1]->triangles()) {
        assert(tri->isValid());
    }


    mesh->validateTriangles();
#endif

    return S_OK;
}

float RadialEdgeSplit::energy() const {
    return -(Magnum::Math::distance(edge[0]->position, edge[1]->position) - longCutoff);
}

bool RadialEdgeSplit::depends(CTrianglePtr tri) const {
    for(int i = 0; i < 3; ++i) {
        if(equals({{tri->vertices[i], tri->vertices[(i+1)%3]}})) {
            return true;
        }
    }
    return false;
}

bool RadialEdgeSplit::depends(CVertexPtr v) const {
    return v == edge[0] || v == edge[1];
}

bool RadialEdgeSplit::equals(const Edge& e) const {
    return (e[0] == edge[0] && e[1] == edge[1]) ||
           (e[0] == edge[1] && e[1] == edge[0]);
}

void RadialEdgeSplit::mark() const {
        std::cout << "marking radial edge split edge {" << edge[0]->id << ", " << edge[1]->id << "}" << std::endl;

    for(TrianglePtr tri : mesh->triangles) {
        tri->color[3] = 0;
        tri->alpha = 0.3;
    }

    EdgeTriangles triangles(edge);

    for(TrianglePtr tri : triangles) {
        tri->color = Magnum::Color4{1, 1, 0, 1};
    }
}

bool RadialEdgeSplit::equals(CVertexPtr) const
{
    return false;
}
