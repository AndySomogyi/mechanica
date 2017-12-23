/*
 * MeshRelationships.cpp
 *
 *  Created on: Sep 29, 2017
 *      Author: andy
 */

#include <MeshRelationships.h>
#include <algorithm>
#include <iostream>

bool incident(CTrianglePtr t, CCellPtr c) {
    return t->cells[0] == c || t->cells[1] == c;
}

bool adjacent_triangle_vertices(CTrianglePtr a, CTrianglePtr b) {

    for(int k = 0; k < 3; ++k) {
        if ((a->vertices[0] == b->vertices[k] &&
             (a->vertices[1] == b->vertices[(k+1)%3] ||
              a->vertices[1] == b->vertices[(k+2)%3] ||
              a->vertices[2] == b->vertices[(k+1)%3] ||
              a->vertices[2] == b->vertices[(k+2)%3])) ||
            (a->vertices[1] == b->vertices[k] &&
             (a->vertices[0] == b->vertices[(k+1)%3] ||
              a->vertices[0] == b->vertices[(k+2)%3] ||
              a->vertices[2] == b->vertices[(k+1)%3] ||
              a->vertices[2] == b->vertices[(k+2)%3])) ||
            (a->vertices[2] == b->vertices[k] &&
             (a->vertices[0] == b->vertices[(k+1)%3] ||
              a->vertices[0] == b->vertices[(k+2)%3] ||
              a->vertices[1] == b->vertices[(k+1)%3] ||
              a->vertices[1] == b->vertices[(k+2)%3]))) {
                 return true;
             }
    }
    return false;
}

bool adjacent(CPTrianglePtr a, CPTrianglePtr b) {
    if (!a || !b || a == b) {
        return false;
    }

    bool result =
        (a->neighbors[0] == b || a->neighbors[1] == b || a->neighbors[2] == b) ||
        (b->neighbors[0] == a || b->neighbors[1] == a || b->neighbors[2] == a);

#ifndef NDEBUG
    if(result) assert(adjacent_triangle_vertices(a->triangle, b->triangle));
#endif
    return result;
}


void connect_triangle_partial_triangles(TrianglePtr a, TrianglePtr b, CCellPtr cell) {
    // check to see that triangles share adjacent vertices.
    assert(adjacent_triangle_vertices(a, b));

#ifndef NDEBUG
    bool found = false;
#endif

    if(cell) {
        // hook up the partial triangles on the correct cell sides.
        for(uint i = 0; i < 2; ++i) {
            for(uint j = 0; j < 2; ++j) {
                if(cell == a->cells[i] && cell == b->cells[j]) {
                    connect_partial_triangles(&a->partialTriangles[i], &b->partialTriangles[j]);
#ifndef NDEBUG
                    found = true;
#endif

                }
            }
        }

    }
    else {
        // hook up the partial triangles on the correct cell sides.
        for(uint i = 0; i < 2; ++i) {
            for(uint j = 0; j < 2; ++j) {
                if(a->cells[i] == b->cells[j]) {
                    connect_partial_triangles(&a->partialTriangles[i], &b->partialTriangles[j]);
#ifndef NDEBUG
                    found = true;
#endif
                }
            }
        }
    }
    assert(found);
}

typedef std::array<int, 2> EdgeIndx;

inline EdgeIndx adjacent_edge_indx(CPTrianglePtr a, CPTrianglePtr b) {
    assert(a->triangle != b->triangle && "partial triangles are on the same triangle");

    EdgeIndx result;

    VertexPtr v1 = nullptr, v2 = nullptr;


    for(int i = 0; i < 3  && (v1 == nullptr || v2 == nullptr); ++i) {
        for(int j = 0; j < 3 && (v1 == nullptr || v2 == nullptr); ++j) {
            if(!v1 && a->triangle->vertices[i] == b->triangle->vertices[j]) {
                v1 = a->triangle->vertices[i];
                continue;
            }
            if(!v2 && a->triangle->vertices[i] == b->triangle->vertices[j]) {
                v2 = a->triangle->vertices[i];
            }
        }
    }

    assert(v1 && v2);
    assert(v1 != v2);

    result[0] = a->triangle->adjacentEdgeIndex(v1, v2);
    result[1] = b->triangle->adjacentEdgeIndex(v1, v2);

    return result;
}

void connect_partial_triangles(PTrianglePtr a, PTrianglePtr b) {

    EdgeIndx edge = adjacent_edge_indx(a, b);

#ifndef NDEBUG
    Edge ea = {{a->triangle->vertices[edge[0]], a->triangle->vertices[(edge[0]+1)%3]}};
    Edge eb = {{b->triangle->vertices[edge[1]], b->triangle->vertices[(edge[1]+1)%3]}};
    assert(incident(a->triangle, ea));
    assert(incident(a->triangle, eb));
    assert(incident(b->triangle, ea));
    assert(incident(b->triangle, eb));
    assert(a->neighbors[edge[0]] == nullptr);
    assert(b->neighbors[edge[1]] == nullptr);
#endif

    a->neighbors[edge[0]] = b;
    b->neighbors[edge[1]] = a;
}

void disconnect_partial_triangles(PTrianglePtr a, PTrianglePtr b) {
    for(int i = 0; i < 3; ++i) {
        if(a->neighbors[i] == b) {
            a->neighbors[i] = nullptr;
        }

        if(b->neighbors[i] == a) {
            b->neighbors[i] = nullptr;
        }
    }
}

bool incident(CTrianglePtr tri, CVertexPtr v) {
    assert(tri);
    return tri->vertices[0] == v || tri->vertices[1] == v || tri->vertices[2] == v;
}

bool incident(CPTrianglePtr pt, const Edge& edge) {
    return incident(pt->triangle, edge);
}

bool incident(CTrianglePtr tri, const std::array<VertexPtr, 2>& edge) {
    return incident(tri, edge[0]) && incident(tri, edge[1]);
}


void reconnect(PTrianglePtr o, PTrianglePtr n, const std::array<VertexPtr, 2>& edge) {
    for(uint i = 0; i < 3; ++i) {
        if(o->neighbors[i] &&
           o->neighbors[i]->triangle &&
           incident(o->neighbors[i]->triangle, edge)) {
            PTrianglePtr adj = o->neighbors[i];
            disconnect_partial_triangles(o, adj);
            connect_partial_triangles(n, adj);
            return;
        }
    }
    assert(0 && "partial triangle is not adjacent to given edge");

}

bool incident(CPTrianglePtr tri, CVertexPtr v) {
    return incident(tri->triangle, v);
}

void disconnect_triangle_vertex(TrianglePtr tri, VertexPtr v) {
    if(!v) return;

    for(uint i = 0; i < 3; ++i) {
        if(tri->vertices[i] == v) {
            v->removeTriangle(tri);
            tri->vertices[i] = nullptr;
            return;
        }
    }

    assert(0 && "triangle did not match vertex");
}

void connect_triangle_vertex(TrianglePtr tri, VertexPtr v) {
    for(int i = 0; i < 3; ++i) {
        if(tri->vertices[i] == nullptr) {
            tri->vertices[i] = v;
            v->appendTriangle(tri);
            return;
        }
    }
    assert(0 && "triangle has no empty slot");
}

int radialedge_connect_triangle(TrianglePtr tri, int edgeIndx) {
    assert(edgeIndx >= 0 && edgeIndx < 3);
    assert(tri->vertices[edgeIndx] && tri->vertices[(edgeIndx+1)%3]);
    assert(tri->cells[0] && tri->cells[1]);

    TrianglePtr ringTri = nullptr;

    for(TrianglePtr t : tri->vertices[edgeIndx]->triangles()) {
        if(incident(t, tri->vertices[(edgeIndx+1)%3])) {
            ringTri = t;
        }
    }

    assert(ringTri);

    TrianglePtr t = ringTri;
    do {

        t = t->edgeRing[edgeIndx];
    } while(t != ringTri);

}

HRESULT replaceTriangleVertex(TrianglePtr tri, VertexPtr o, VertexPtr v) {
    for(int i = 0; i < 3; ++i) {
        if(tri->vertices[i] == o) {
            HRESULT result = o->removeTriangle(tri);
            tri->vertices[i] = v;
            result |= v->appendTriangle(tri);
            return result;
        }
    }
    return E_FAIL;
}

int radialedge_disconnect_triangle(TrianglePtr tri, int edgeIndex) {
}

bool adjacent(CVertexPtr v1, CVertexPtr v2) {
    if(v1->triangles().size() < v2->triangles().size()) {
        for(TrianglePtr tri : v1->triangles()) {
            if(incident(tri, v2)) {
                return true;
            }
        }
    } else {
        for(TrianglePtr tri : v2->triangles()) {
            if(incident(tri, v1)) {
                return true;
            }
        }
    }
    return false;
}

bool adjacent_triangle_pointers(CTrianglePtr a, CTrianglePtr b)
{
    for(int i = 0; i < 2; ++i) {
        for(int j = 0; j < 3; ++j) {
            if(a->partialTriangles[i].neighbors[j] &&
               a->partialTriangles[i].neighbors[j]->triangle == b) {
                return true;
            }
        }
    }
    return false;
}
