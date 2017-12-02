/*
 * MeshRelationships.cpp
 *
 *  Created on: Sep 29, 2017
 *      Author: andy
 */

#include <MeshRelationships.h>
#include <algorithm>

bool incident(CTrianglePtr t, CCellPtr c) {
    return t->cells[0] == c || t->cells[1] == c;
}

bool adjacent(CTrianglePtr a, CTrianglePtr b) {
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

bool incident(CFacetPtr facet, CCellPtr cell) {
    assert(facet);
    return facet && (facet->cells[0] == cell || facet->cells[1] == cell);
}

bool adjacent(CPTrianglePtr a, CPTrianglePtr b) {
    if (!a || !b || a == b) {
        return false;
    }

    bool result =
        (a->neighbors[0] == b || a->neighbors[1] == b || a->neighbors[2] == b) ||
        (b->neighbors[0] == a || b->neighbors[1] == a || b->neighbors[2] == a);

#ifndef NDEBUG
    if(result) assert(adjacent(a->triangle, b->triangle));
#endif
    return result;
}

//bool incident(CVertexPtr vertex, CFacetPtr facet) {
//    return contains(vertex->facets(), facet);
//}

void connect_triangle_partial_triangles(TrianglePtr a, TrianglePtr b) {
    // check to see that triangles share adjacent vertices.
    assert(adjacent(a, b));

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
