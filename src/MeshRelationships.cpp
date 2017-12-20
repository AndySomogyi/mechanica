/*
 * MeshRelationships.cpp
 *
 *  Created on: Sep 29, 2017
 *      Author: andy
 */

#include <MeshRelationships.h>
#include <algorithm>

/**
 * Replace an adjacent triangle of tri at index edgeIndx with n.
 *
 * If the triangle slot is not-empty, the triangle that it points to
 * gets it's neighbor slots set to null.
 *
 * If the base triangle, tri is already attached to an existing
 * triangle where the new triangle will go, there are two posibilities.
 *
 * a: this is a manifold edge, where only two triangles share the edge.
 *
 * b: this is a radial edge, where more than two triangles share the edge.
 */
static int connect_triangle(TrianglePtr tri,  int edge, int tcell, int ncell, TrianglePtr n) {
    int result = 0;
    if(tri->adjTriangles[tcell][edge]) {
        TrianglePtr o = tri->adjTriangles[tcell][edge];
        VertexPtr v1 = tri->vertices[edge];
        VertexPtr v2 = tri->vertices[(edge+1)%3];
        int oi = o->adjacentEdgeIndex(v1, v2);
        assert(oi >= 0);
        int ocell = (o->cells[0] == tri->cells[tcell]) ? 0 : 1;

        TrianglePtr ot = tri->partialTriangles[tcell].neighbors[edge]->triangle;
        assert(ot == o);
        assert(o->adjTriangles[ocell][oi] == tri);
        assert(o->partialTriangles[ocell].neighbors[oi]->triangle == tri);
        bool b1 = o->partialTriangles[ocell].neighbors[oi] == &tri->partialTriangles[0];
        bool b2 = o->partialTriangles[ocell].neighbors[oi] == &tri->partialTriangles[1];
        assert(o->partialTriangles[ocell].neighbors[oi] == &tri->partialTriangles[tcell]);
        o->adjTriangles[ocell][oi] = nullptr;
        o->partialTriangles[ocell].neighbors[oi] = nullptr;
        result = 1;
    }
    tri->adjTriangles[tcell][edge] = n;
    tri->partialTriangles[tcell].neighbors[edge] = &n->partialTriangles[ncell];
    return result;
}


bool incident(CTrianglePtr t, CCellPtr c) {
    return t->cells[0] == c || t->cells[1] == c;
}

bool adjacent_vertices(CTrianglePtr a, CTrianglePtr b) {
    if(a == b) {
        return false;
    }

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
    return adjacent_vertices(a->triangle, b->triangle);
}


void connect_triangle_partial_triangles(TrianglePtr a, TrianglePtr b) {
    // check to see that triangles share adjacent vertices.
    assert(adjacent_vertices(a, b));

    #ifndef NDEBUG
    int conCnt = 0;
    int rootCnt = 0;
    #endif

    // hook up the partial triangles on the correct cell sides.
    for(uint i = 0; i < 2; ++i) {
        // don't connect root facing ptris.
        // in non-debug mode, we never hit the inner loop if
        // a[i] is the root cell.
        #ifdef NDEBUG
        if(a->cells[i]->isRoot()) continue;
        #endif

        for(uint j = 0; j < 2; ++j) {
            if(a->cells[i] == b->cells[j]) {
                #ifndef NDEBUG
                if(a->cells[i]->isRoot()) {
                    rootCnt++;
                    continue;
                }
                #endif
                connect_partial_triangles(&a->partialTriangles[i], &b->partialTriangles[j]);
                #ifndef NDEBUG
                conCnt++;
                #endif
            }
        }
    }

    #ifndef NDEBUG
    assert(rootCnt > 0 || conCnt > 0);
    #endif
}

void connect_partial_triangles(PTrianglePtr a, PTrianglePtr b) {
    assert(a->triangle != b->triangle && "partial triangles are on the same triangle");

#ifdef NEW_TRIANGLE_ADJ

    int result = connect_triangles(a->triangle, b->triangle);
    assert(result >= 0);

#else

    assert((!a->neighbors[0] || !a->neighbors[1] || !a->neighbors[2])
           && "connecting partial face without empty slots");
    assert((!b->neighbors[0] || !b->neighbors[1] || !b->neighbors[2])
           && "connecting partial face without empty slots");

    for(uint i = 0; i < 3; ++i) {
        assert(a->neighbors[i] != a && a->neighbors[i] != b);
        if(!a->neighbors[i]) {
            a->neighbors[i] = b;
            break;
        }
    }

    for(uint i = 0; i < 3; ++i) {
        assert(b->neighbors[i] != a && b->neighbors[i] != b);
        if(!b->neighbors[i]) {
            b->neighbors[i] = a;
            break;
        }
    }
#endif
}

void disconnect_partial_triangles(PTrianglePtr a, PTrianglePtr b) {
    for(uint i = 0; i < 3; ++i) {
        if(a->neighbors[i] == b) {
            a->neighbors[i] = nullptr;
            break;
        }
    }

    for(uint i = 0; i < 3; ++i) {
        if(b->neighbors[i] == a) {
            b->neighbors[i] = nullptr;
            break;
        }
    }
}

bool incident(CTrianglePtr tri, CVertexPtr v) {
    assert(tri);
    return tri->vertices[0] == v || tri->vertices[1] == v || tri->vertices[2] == v;
}

/*
void disconnect(TrianglePtr tri, const Edge& edge) {
    assert(incident(tri, edge[0]));
    assert(incident(tri, edge[1]));
    if(!tri->cells[0]->isRoot()) { disconnect(&tri->partialTriangles[0], edge); }
    if(!tri->cells[1]->isRoot()) { disconnect(&tri->partialTriangles[1], edge); }
}

void disconnect(PTrianglePtr pt, const Edge& edge) {
    for(uint i = 0; i < 3; ++i) {
        if(pt->neighbors[i] &&
           pt->neighbors[i]->triangle &&
           incident(pt->neighbors[i]->triangle, edge[0]) &&
           incident(pt->neighbors[i]->triangle, edge[1])) {
            disconnect(pt, pt->neighbors[i]);
            return;
        }
    }
    assert(0 && "partial triangle is not adjacent to given edge");
}
*/

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

void reconnect_triangles(TrianglePtr o, TrianglePtr n, const std::array<VertexPtr, 2>& edge) {

    int oi = o->adjacentEdgeIndex(edge[0], edge[1]);
    int ni = n->adjacentEdgeIndex(edge[0], edge[1]);

    assert(oi >= 0 && ni >= 0);

    /*

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
    */

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

    /*
    for(uint i = 0; i < 2; ++i) {
        for(uint j = 0; j < 3; ++j) {
            if(tri->partialTriangles[i].neighbors[j] &&
                    incident(tri->partialTriangles[i].neighbors[j], v)) {
                disconnect(&tri->partialTriangles[i], tri->partialTriangles[i].neighbors[j]);
            }
        }
    }
    */
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


    //for(TrianglePtr t = ringTri; t )







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

static inline bool _directional_adjacent_pointers(CTrianglePtr a, CTrianglePtr b, int cellIndex) {
    for(int i = 0; i < 3; ++i) {
        if(a->adjTriangles[cellIndex][i] == b) {
            return true;
        }
    }
    return false;
}

bool adjacent_pointers(CTrianglePtr a, CTrianglePtr b, int cellIndex) {
    bool result;
    if(cellIndex == 0) {
        result = _directional_adjacent_pointers(a, b, 0);
#ifndef NDEBUG
        assert(result == _directional_adjacent_pointers(b, a, 0));
#endif
        return result;
    }
    else if(cellIndex == 1) {
        result = _directional_adjacent_pointers(a, b, 1);
#ifndef NDEBUG
        assert(result == _directional_adjacent_pointers(b, a, 1));
#endif
        return result;
    }
    else {
#ifndef NDEBUG
        return _directional_adjacent_pointers(a, b, 0) ||
            _directional_adjacent_pointers(a, b, 1);
#else
        bool r0 = _directional_adjacent_pointers(a, b, 0);
        assert(r0 == _directional_adjacent_pointers(b, a, 0));
        
        bool r1 = _directional_adjacent_pointers(a, b, 1);
        assert(r1 == _directional_adjacent_pointers(b, a, 1));
        
        return r0 || r1;
#endif
    }
}

int connect_triangles(TrianglePtr a, TrianglePtr b) {
    // the shared vertices
    VertexPtr v1 = nullptr, v2 = nullptr;

    int result = -1;

    for(int i = 0; i < 3; ++i) {
        for(int j = 0; j < 3; ++j) {
            if(!v1 && a->vertices[i] == b->vertices[j]) {
                v1 = a->vertices[i];
            }
            else if(!v2 && a->vertices[i] == b->vertices[j]) {
                v2 = a->vertices[i];
            }
            if(v1 && v2) { break; }
        }
        if(v1 && v2) { break; }
    }

    if(!v1 || !v2) { return -1; }

    assert(v1 != v2);

    int ai = a->adjacentEdgeIndex(v1, v2);

    int bi = b->adjacentEdgeIndex(v1, v2);

    assert(ai >= 0 && bi >= 0);

    if(a->cells[0] == b->cells[0]) {
        result = connect_triangle(a, ai, 0, 0, b);
        result |= connect_triangle(b, bi, 0, 0, a);
        assert(a->cells[0] != b->cells[1]);
        assert(a->cells[1] != b->cells[0]);
    }
    if(a->cells[0] == b->cells[1]) {
        result = connect_triangle(a, ai, 0, 1, b);
        result |= connect_triangle(b, bi, 1, 0, a);
        assert(a->cells[0] != b->cells[0]);
        assert(a->cells[1] != b->cells[1]);
    }
    if(a->cells[1] == b->cells[0]) {
        result = connect_triangle(a, ai, 1, 0, b);
        result |= connect_triangle(b, bi, 0, 1, a);
        assert(a->cells[0] != b->cells[0]);
        assert(a->cells[1] != b->cells[1]);
    }
    if(a->cells[1] == b->cells[1]) {
        result = connect_triangle(a, ai, 1, 1, b);
        result |= connect_triangle(b, bi, 1, 1, a);
        assert(a->cells[0] != b->cells[1]);
        assert(a->cells[1] != b->cells[0]);
    }

    return result;
}

HRESULT disconnect_triangle_from_cell(TrianglePtr tri, CCellPtr cell) {
}

HRESULT disconnect_triangles(TrianglePtr a, TrianglePtr b, int cell) {
}
