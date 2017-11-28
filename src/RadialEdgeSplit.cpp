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
    EdgeTriangles e{edge};

    std::vector<TrianglePtr> triangles(e.begin(), e.end());

    assert(triangles.size() >= 2);

#ifndef NDEBUG
    std::vector<TrianglePtr> newTriangles;
#endif

    ctr += 1;

    // new vertex at the center of this edge
    Vector3 center = (edge[0]->position + edge[1]->position) / 2.;
    center = center + (edge[1]->position - edge[1]->position) * uniformDist(randEngine);
    VertexPtr vert = mesh->createVertex(center);

    TrianglePtr firstNewTri = nullptr;
    TrianglePtr prevNewTri = nullptr;

    for(uint i = 0; i < triangles.size(); ++i)
    {
        TrianglePtr tri = triangles[i];
        std::cout << "Radial Edge Split: tri[" << i << "], cell[0]:" << tri->cells[0] << ", cell[1]:" << tri->cells[1] << std::endl;
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


        if(tri->facet) {
            tri->facet->appendChild(nt);
        }

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
        assert(adjacent(tri, nt));

        // removes the edge[1] - outer edge connection connection from the old
        // triangle and replaces it with the new triangle,
        // manually add the partial triangles to the cell
        for(uint i = 0; i < 2; ++i) {
            if(tri->cells[i] != mesh->rootCell()) {

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
        }

        assert(incident(nt, {{edge[1], outer}}));
        assert(!incident(tri, {{edge[1], outer}}));


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
            connect_triangle_partial_triangles(nt, prevNewTri);
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
        connect_triangle_partial_triangles(firstNewTri, prevNewTri);
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
