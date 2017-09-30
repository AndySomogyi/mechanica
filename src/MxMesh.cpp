/*
 * MxMesh.cpp
 *
 *  Created on: Jan 31, 2017
 *      Author: andy
 */


#include "MxDebug.h"
#include <MxMesh.h>
#include <Magnum/Math/Math.h>
#include "MagnumExternal/Optional/optional.hpp"

#include <deque>
#include <limits>



int MxMesh::findVertex(const Magnum::Vector3& pos, double tolerance) {
    for (int i = 1; i < vertices.size(); ++i) {
        float dist = (vertices[i].position - pos).dot();
        if (dist <= tolerance) {
            return i;
        }
    }
    return -1;
}

VertexPtr MxMesh::createVertex(const Magnum::Vector3& pos) {
    vertices.push_back({pos, {}, {}});
    return &vertices[vertices.size() - 1];
}

MxCell& MxMesh::createCell(MxCellType *type) {
    cells.push_back(MxCell{type, this, nullptr});
    return cells[cells.size() - 1];
}

void MxMesh::vertexAtributes(const std::vector<MxVertexAttribute>& attributes,
        uint vertexCount, uint stride, void* buffer) {
}


void MxMesh::dump(uint what) {
    for(int i = 0; i < vertices.size(); ++i) {
        std::cout << "[" << i << "]" << vertices[i].position << std::endl;
    }
}

#include <random>

std::default_random_engine eng;

void MxMesh::jiggle() {

    std::uniform_real_distribution<float> distribution(-0.002,0.002);

    for (int i = 0; i < vertices.size(); ++i) {

        Vector3 test = vertices[i].position + Vector3{distribution(eng), distribution(eng), distribution(eng)};

        if((test - initPos[i]).length() < 0.7) {
            vertices[i].position  = test;
        }
    }
}

std::tuple<Magnum::Vector3, Magnum::Vector3> MxMesh::extents() {

    auto min = Vector3{std::numeric_limits<float>::max()};
    auto max = Vector3{std::numeric_limits<float>::min()};


    for(auto& v : vertices) {
        for(int i = 0; i < 3; ++i) {min[i] = (v.position[i] < min[i] ? v.position[i] : min[i]);}
        for(int i = 0; i < 3; ++i) {max[i] = (v.position[i] > max[i] ? v.position[i] : max[i]);}
    }

    return std::make_tuple(min, max);
}

TrianglePtr MxMesh::createTriangle(const std::array<VertexPtr, 3> &vertInd) {
    for (MxTriangle& tri : triangles) {
        if (tri.matchVertexIndices(vertInd) != 0) {
            return &tri;
        }
    }

    MxTriangle tri;
    tri.vertices = vertInd;

    triangles.push_back(tri);
    return &triangles[triangles.size() - 1];
}






MxCellType universeCellType = {};

MxCellType *MxUniverseCell_Type = &universeCellType;

MxPartialTriangleType universePartialTriangleType = {};

MxPartialTriangleType *MxUniversePartialTriangle_Type =
        &universePartialTriangleType;

MxMesh::MxMesh()
{
    createCell(MxUniverseCell_Type);
    vertices.push_back(MxVertex{});
}




/**
 * Configuration only valid for 5 cell arrangement, two cells on each face of triangle,
 * and one cell attached to each side.
 */
HRESULT MxMesh::collapseHTriangleOld(MxTriangle& tri) {

    struct CellTriangles {
        MxCell *cell;
        // the pair of triangles attached to the original triangle that will be
        // removed and re-attached.
        std::array<MxTriangle*, 2> rtri;
        // the pair of triangles on the facets of the side cells that will be split,
        // and the removed triangles will be inserted between these. These two triangles
        // are on the right face of the cell, i.e. in the faces array below, the triangles
        // in faces[i].stri are between the faces[i].cell and faces[(i+1)%3].cell
        std::array<MxTriangle*, 2> stri;
    };

    // Get a pair of triangles attached to an edge. Order them such that
    // result[0] is the top triangle, one facing tri.cells[0], and
    // result[1] is the bottom triangle facing tri.cells[1]
    auto incidentCell = [this, tri](const std::array<VertexPtr, 2> &edge) -> std::optional<CellTriangles> {
        auto triangles = EdgeTriangles(this, tri, edge);
        int i = 0;
        CellTriangles result;

        for(MxTriangle &t : triangles) {

            switch(i) {
            case 0:
                continue;
            case 1:
                if(t.cells[0] == tri.cells[0] &&
                   t.cells[1] != tri.cells[0] &&
                   t.cells[1] != tri.cells[1]) {
                       result.cell = t.cells[1];
                       result.rtri[0] = &t;
                }
                else if(t.cells[0] == tri.cells[1] &&
                   t.cells[1] != tri.cells[1] &&
                   t.cells[1] != tri.cells[0]) {
                       result.cell = t.cells[1];
                       result.rtri[0] = &t;
                 }
                else if(t.cells[1] == tri.cells[0] &&
                   t.cells[0] != tri.cells[0] &&
                   t.cells[0] != tri.cells[1]) {
                       result.cell = t.cells[0];
                       result.rtri[0] = &t;
                }
                else if(t.cells[1] == tri.cells[1] &&
                   t.cells[0] != tri.cells[1] &&
                   t.cells[0] != tri.cells[0]) {
                       result.cell = t.cells[0];
                       result.rtri[0] = &t;
                 }
                else {
                    return {};
                }
                break;
            case 2:
                // make sure that the second triangle references the correct cells
                if((t.cells[0] == result.cell &&
                        (t.cells[1] == tri.cells[0] || t.cells[1] == tri.cells[1])) ||
                   (t.cells[1] == result.cell &&
                        (t.cells[0] == tri.cells[0] || t.cells[0] == tri.cells[1]))) {
                    result.rtri[1] = &t;
                    return {result};
                }
                else {
                    return {};
                }
                break;
            default:
                // gone past 2 triangles, not valid configuration
                return {};
            }
            i++;
        }
        return {};
    };




    // the 6 surrounding triangles. The 3 ones facing the top 0 cell are in the
    // faces[x][0] spot, and the bottom ones are in faces[x][1] spot.
    CellTriangles faces[3] = {0, 0, 0};

    // i = {0,1,2}, j = {1,2 0}
    for(int i = 0; i < 3; ++i)
    {
        std::optional<CellTriangles> t = incidentCell({tri.vertices[i], tri.vertices[(i+1)%3]});
        if(t) {
            faces[i] = *t;
        } else {
        		// TODO, raise an error here
            return false;
        }
    }


    // at this point, we've verified that we have 5 cells facing this triangle, we assume
    // that the initial candidate triangle has unique cells on each face.

    // two new vertices, positions adjusted
    MxVertex *verts[2] = {createVertex({0., 0., 0.}), createVertex({0., 0., 0.})};


    auto findSplitTriangles = [this, tri, verts](VertexPtr splitVert,
            CellPtr c1, CellPtr c2) -> std::array<TrianglePtr, 2> {
        struct TriPosn {TrianglePtr tri; std::array<Vector3*, 3> posns;};
        std::vector<TriPosn> triPosns;
    };

    // find the triangles in the side faces to split
    for(int i = 0; i < 3; ++i) {

    }

    auto findEdge = [this, tri](const MxTriangle &t) -> MxEdge {
        VertexPtr verts[2];
        int found = 0;

        for(int i = 0; i < 3 && found < 2; ++i) {
            for(int j = 0; j < 3 && found < 2; ++j) {
                if(t.vertices[i] == tri.vertices[j]) {
                    verts[found++] = t.vertices[i];
                }
            }
        }
        assert(found == 2);
        return MxEdge{verts[0], verts[1]};
    };

    // go through the triangles to be removed from the upper and lower faces, and
    // reconnected to the side faces. this loop reconnects the partial face indices
    // both the upper and lower triangle share the same edge with the center tri.
    for(int i = 0; i < 3; ++i) {
        MxEdge edge = findEdge(*faces[i].rtri[0]);
        triangleManifoldDisconnect(*faces[i].rtri[0], edge);
        triangleManifoldDisconnect(*faces[i].rtri[1], edge);
    }

    // go over all of the triangles attached to the existing three vertices of the
    // triangle to be removed. Re-attach these triangles to either the new top
    // vertex, or the new bottom vertex, depending on if the triangle is in the
    // top or bottom cell, i.e shares a face with the top or bottom cells.
    for(int i = 0; i < 3; ++i) {


        // set of vertices attached to the triangle vertex
        for (auto t : tri.vertices[i]->triangles) {
            // attach to new upper vertex
            if (t->cells[0] == tri.cells[0] || t->cells[1] == tri.cells[0]) {
                for(int j = 0; i < 3; ++j) {
                    if (t->vertices[j] == tri.vertices[i]) {
                        t->vertices[j] = verts[0];
                        break;
                    }
                }
            }
            // attach to lower vertex
            else {
                for(int j = 0; i < 3; ++j) {
                    if (t->vertices[j] == tri.vertices[i]) {
                        t->vertices[j] = verts[1];
                        break;
                    }
                }
            }
        }
    }

    // re-use the adjacent triangles, we 'flip' them sideways, and insert them into the
    // side faces.

    // done re-attaching adjacent triangles, now finish up by deleting
    // the triangle, and it's vertices.
    return true;
}

HRESULT MxMesh::collapseIEdge(MxEdge& edge) {
	return 1;
}

void MxMesh::vertexReconnect(VertexPtr o, VertexPtr n) {
}

void MxMesh::triangleManifoldDisconnect(const MxTriangle& tri,
        const MxEdge& edge) {
}

bool MxMesh::splitWedgeVertex(VertexPtr v0, VertexPtr nv0, VertexPtr nv1,
        MxCell* c0, MxCell* c1, MxTriangle* tri) {

    struct TriEdge {MxTriangle *t0; MxTriangle *t1; float d0; float d1; };
    std::deque<TriEdge> wedge;

    MxTriangle *startTri = 0;

    Vector3 v0_pos = vertex(v0).position;
    Vector3 nv0_pos = vertex(nv0).position;
    Vector3 nv1_pos = vertex(nv1).position;

    for(TrianglePtr t : v0->triangles) {
        if (incident(t, c0) && incident(t, c1)) {
            startTri = t;
            break;
        }
    }

    if (startTri == 0) {
        return false;
    }

    auto triDistance = [this, &v0_pos, v0] (const MxTriangle *t) {
        Vector3 p1, p2;
        if(t->vertices[0] != v0) {
            p1 = vertex(t->vertices[1]).position;
            p2 = vertex(t->vertices[2]).position;
        }
        else if(t->vertices[1] != v0) {
            p1 = vertex(t->vertices[0]).position;
            p2 = vertex(t->vertices[2]).position;
        }
        else {
            p1 = vertex(t->vertices[0]).position;
            p2 = vertex(t->vertices[1]).position;
        }
        (p1 + p2) / 2.;
        return 0;
    };

    auto nextTri = [this, v0] (const MxTriangle *prev, const MxTriangle *current, MxCell *c) -> MxTriangle* {
        auto &p = current->partialTriangles[0];
        MxTriangle *n = p.neighbors[0]->triangle;
        if(n != prev && incident(n, v0) && incident(n, c)) {
            return n;
        }
        n = p.neighbors[1]->triangle;
        if(n != prev && incident(n, v0) && incident(n, c)) {
            return n;
        }
        n = p.neighbors[2]->triangle;
        if(n != prev && incident(n, v0) && incident(n, c)) {
            return n;
        }
        return nullptr;
    };



    // Grab all of the triangle pointers from the wedge, add them to a
    // container (wedge). Have to do squirrelly iteration with prev pointers
    // to keep track where to go next.
    MxTriangle *prev = nextTri(nullptr, startTri, c1);
    MxTriangle *next = nextTri(prev, startTri, c1);
    MxTriangle *curr = startTri;

    // iterate over the list backwards until we find the start of the fan.
    while(prev != nullptr) {
        wedge.push_front({prev, curr, 0, 0});
        MxTriangle *tmp = nextTri(curr, prev, c1);
        curr = prev;
        prev = tmp;
    }

    // grab the triangle before the wedge (not part of the wedge, but before),
    // don't check the opposite side faces the opposite cell
    wedge.push_front({nextTri(wedge.front().t1, wedge.front().t0, c0), wedge.front().t0, 0, 0});

    // iterate over the fan forward until we find the end of the fan.
    curr = startTri;
    while(next != nullptr) {
        wedge.push_back({curr, next, 0, 0});
        MxTriangle *tmp = nextTri(curr, next, c1);
        curr = next;
        next = tmp;
    }

    // grab the triangle after the wedge
    wedge.push_back({nextTri(wedge.back().t0, wedge.back().t1, c0), wedge.back().t1, 0, 0});

#ifndef NDEBUG
    for(int i = 0; i < wedge.size() - 1; ++i) {
        TriEdge &e = wedge[i];
        assert(adjacent(e.t0, e.t1));
        assert(wedge[i].t1 == wedge[i+1].t0);
    }
#endif

    // find the edge with the smallest distance difference to the
    // new vertex positions
    float min = std::numeric_limits<float>::max();

    // the triangles that will be split apart and the
    // new triangle inserted between them
    MxTriangle *left = nullptr, *right = nullptr;

    // shared far vertex, above triangles will remain
    // connected on the far side.
    VertexPtr v;

    for(auto &e : wedge) {
        prev = e.t0;
        next = e.t1;

        // get the shared far vertex
        if (prev->vertices[0] != v0 && incident(next, prev->vertices[0])) {
            v = prev->vertices[0];
        }
        else if (prev->vertices[1] != v0 && incident(next, prev->vertices[1])) {
            v = prev->vertices[1];
        }
        else {
            assert(prev->vertices[2] != v0 && incident(next, prev->vertices[2]));
            v = prev->vertices[2];
        }

        Vector3 pos = vertex(v).position;

        e.d0 = (nv0_pos - pos).length();
        e.d1 = (nv1_pos - pos).length();
        float d = std::abs(e.d0 - e.d1);
        if (d < min) {
            min = d;
            left = prev;
            right = next;
        }
    }

    assert(left && right);

    // we now have the left and right triangles, where the new triangle
    // will be inserted between.
    for(auto &e : wedge) {
        if(e.t0 == left && e.t1 == right) {

        }
        else {

        }
    }



    return false;
}

/*
MxEdge MxMesh::edge(const MxTriangle& t1, MxTriangle& t2) {
    if(t1.incident(t2.vertices[0])) {
        if(t1.incident(t2.vertices[1])) {
            return MxEdge{t2.vertices[0], t2.vertices[1]};
        }
        if(t1.incident(t2.vertices[2])) {
            return MxEdge{t2.vertices[0], t2.vertices[2]};
        }
    }
    else if(t1.incident(t2.vertices[1]) && t1.incident(t2.vertices[2])) {
        return MxEdge{t2.vertices[1], t2.vertices[1]};
    }
    return MxEdge{0,0};
}
*/


/**
 * Determine what kind of edge we have.
 */
HRESULT MxMesh::collapseEdge(MxEdge& edge) {

	// check if we have a manifold edge, most common kind of short edge
	if (edge.upperFacets().size() == 0 &&
	    edge.lowerFacets().size() == 0 &&
		edge.radialFacets().size() == 1) {
		return collapseManifoldEdge(edge);
	}

	if (edge.upperFacets().size() == 1 && edge.upperFacets().size() == 1) {

		// if any one of the facets only has one triangle, that means that it's
		// an H configuration, so collapse that triangle. Note, that means that
		// that operation will also collapse every other triangle that is adjacent
		// to that triangle.
		for(auto f : edge.radialFacets()) {
			if(f->triangles.size() == 1) {
				MxTriangle *t = f->triangles[0];
				return collapseHTriangle(t);
			}
		}

		// at this point, we know that the edge is not manifold
		// is incident to at at least two faces, that means that it's
		// an I configuration, so collapse that.
		return collapseIEdge(edge);
	}

	// could not collapse the edge
	return -1;
}

HRESULT MxMesh::collapseManifoldEdge(MxEdge& e) {
	return -1;
}

/**
 * go around the ring of the edge, and split every incident triangle on
 * that edge. Creates a new vertex at the midpoint of this edge.
 */
HRESULT MxMesh::splitEdge(MxEdge& e) {
	auto triangles = e.radialTriangles();

	for (auto tri : triangles) {

	}

	return 0;
}

HRESULT MxMesh::collapseHTriangle(TrianglePtr tri) {

	// perpendicular facets
	FacetPtr perpFacets[3] = {nullptr};

	// identify the triangle radial facets, these are the facets that are
	// the side facets. We identify the side facets, as those that are
	// incident to a triangle vertex, but not incident to any other
	// triangle vertices. Each triangle vertex should have exactly one
	// Indecent facet that is not indecent to any other triangle vertices
	// for this to be a valid H configuration.
	for(int i = 0; i < 3; ++i) {
		for(FacetPtr p : tri->vertices[i]->facets) {

			bool found = false;

			if(p == tri->facet) {
				continue;
			}

			for(FacetPtr x : tri->vertices[(i+1)%3]->facets) {
				if (x == p) {
					found = true;
					break;
				}
			}

			if (!found) {
				for(FacetPtr x : tri->vertices[(i+2)%3]->facets) {
					if (x == p) {
						found = true;
						break;
					}
				}
			}

			// we did not find this facet is incident to the other two vertices.
			// need to check that that we have not already found a perpendicular facet
			// for this triangle vertex.
			if (!found) {
				if(perpFacets[i] == nullptr) {
					perpFacets[i] = p;
				} else {
					// not a valid H configuration,
					// TODO set some sort of error conditions
					return -1;
				}
			}
		}

		// make sure we've found a perpendicular facet
		assert(perpFacets[i]);
	}

	VertexPtr v0 = collapseCellSingularFacet(tri->cells[0], tri->facet);
	VertexPtr v1 = collapseCellSingularFacet(tri->cells[1], tri->facet);

	TrianglePtr perpTriangles[3];

	for(int i = 0; i < 3; ++i) {
		perpTriangles[i] = splitFacetBoundaryVertex(perpFacets[i], tri->vertices[i], v0, v1);
	}

	return 0;
}



TrianglePtr MxMesh::splitFacetBoundaryVertex(FacetPtr face, VertexPtr v,
		VertexPtr v0, VertexPtr v1) {
}

VertexPtr MxMesh::collapseCellSingularFacet(CellPtr cell, FacetPtr facet) {
}

void MxMesh::triangleVertexReconnect(MxTriangle& tri, VertexPtr o,
		VertexPtr n) {
}
