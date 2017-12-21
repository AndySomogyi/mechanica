/*
 * MxCell.cpp
 *
 *  Created on: Jul 7, 2017
 *      Author: andy
 */


#include <MxCell.h>
#include <MxMesh.h>
#include <iostream>
#include <algorithm>
#include <array>


#include "MxDebug.h"
#include "DifferentialGeometry.h"
#include "MxMeshRenderer.h"

bool operator == (const std::array<MxVertex *, 3>& a, const std::array<MxVertex *, 3>& b) {
  return a[0] == b[0] && a[1] == b[1] && a[2] == b[2];
}


bool MxCell::manifold() const {

    for(auto t : boundary) {

        // check pointers
        if (!adjacent(t, t->neighbors[0]) ||
            !adjacent(t, t->neighbors[1]) ||
            !adjacent(t, t->neighbors[2])) {
            return false;
        }

        // check vertices
        if (!adjacent_triangle_vertices(t->triangle, t->neighbors[0]->triangle) ||
            !adjacent_triangle_vertices(t->triangle, t->neighbors[1]->triangle) ||
            !adjacent_triangle_vertices(t->triangle, t->neighbors[2]->triangle)) {
            return false;
        }
    }
    return true;
}

void MxCell::vertexAtributeData(const std::vector<MxVertexAttribute>& attributes,
        uint vertexCount, uint stride, void* buffer) {
    MxCellType *type = (MxCellType*)ob_type;
    uchar *ptr = (uchar*)buffer;
    for(uint i = 0; i < boundary.size() && ptr < vertexCount * stride + (uchar*)buffer; ++i, ptr += 3 * stride) {


        const MxTriangle &tri = *(boundary[i])->triangle;
        VertexAttribute *attrs = (VertexAttribute*)ptr;

        if(true) {
            for(int i = 0; i < 3; ++i) {
                attrs[i].color = jetColorMap(
                    tri.vertices[i]->attr,
                    MxVertex::minForceDivergence,
                    MxVertex::maxForceDivergence
                );
                attrs[i].color[3] = tri.alpha;
            }
        }
        
        else {
            Color4 color;
            if(tri.color[3] > 0) {
                color = tri.color;
            }
            else if (type) {
                color = type->color;
                color[3] = tri.alpha;
            }
            else {
                color = Color4::yellow();
            }
            attrs[0].color = color;
            attrs[1].color = color;
            attrs[2].color = color;
        }
        
        attrs[0].position = tri.vertices[0]->position;
        attrs[1].position = tri.vertices[1]->position;
        attrs[2].position = tri.vertices[2]->position;
        

    }
}




//void MxCell::indexData(uint indexCount, uint* buffer) {
//    for(int i = 0; i < boundary.size(); ++i, buffer += 3) {
//        const MxTriangle &face = *boundary[i]->triangle;
//        buffer[0] = face.vertices[0];
//        buffer[1] = face.vertices[1];
//        buffer[2] = face.vertices[2];
//    }
//}





void MxCell::dump() {

    for (uint i = 0; i < boundary.size(); ++i) {
        const MxTriangle &ti = *boundary[i]->triangle;

        std::cout << "face[" << i << "] {" << std::endl;
        //std::cout << "vertices:" << ti.vertices << std::endl;
        std::cout << "}" << std::endl;
    }

}

HRESULT MxCell::removeChild(TrianglePtr tri) {
    if(tri->cells[0] != this && tri->cells[1] != this) {
        return mx_error(E_FAIL, "triangle does not belong to this cell");
    }

    int index = tri->cells[0] == this ? 0 : 1;

    tri->cells[index] = nullptr;

    PTrianglePtr pt = &tri->partialTriangles[index];

    for(int i = 0; i < 3; ++i) {
        for(int j = 0; j < 3; ++j) {
            if(pt->neighbors[i] && pt->neighbors[i]->neighbors[j] == pt) {
                pt->neighbors[i]->neighbors[j] = nullptr;
                break;
            }
        }
        pt->neighbors[i] = nullptr;
    }

    remove(boundary, pt);

    return S_OK;
}



/* POV mesh format:
mesh2 {
vertex_vectors {
22
,<-1.52898,-1.23515,-0.497254>
,<-2.41157,-0.870689,0.048214>
,<-3.10255,-0.606757,-0.378837>
     ...
,<-2.22371,-1.5823,-2.07175>
,<-2.41157,-1.30442,-2.18785>
}
face_indices {
40
,<1,4,17>
,<1,17,18>
,<1,18,2>
  ...
,<8,21,20>
,<8,20,9>
}
inside_vector <0,0,1>
}
 */

void MxCell::writePOV(std::ostream& out) {
    out << "mesh2 {" << std::endl;
    out << "face_indices {" << std::endl;
    out << boundary.size()  << std::endl;
    for (int i = 0; i < boundary.size(); ++i) {
        const MxTriangle &face = *boundary[i]->triangle;
        //out << face.vertices << std::endl;
        //auto &pf = boundary[i];
        //for (int j = 0; j < 3; ++j) {
            //Vector3& vert = mesh->vertices[pf.vertices[j]].position;
        //        out << ",<" << vert[0] << "," << vert[1] << "," << vert[2] << ">" << std::endl;
        //}
    }
    out << "}" << std::endl;
    out << "}" << std::endl;
}

HRESULT MxCell::appendChild(TrianglePtr tri, int index) {

    if(tri->cells[0] == this || tri->cells[1] == this) {
        return mx_error(E_FAIL, "triangle is already attached to this cell");
    }

    if (index != 0 && index != 1) {
        return mx_error(E_FAIL, "invalid index argument");
    }

    if (tri->cells[index] != nullptr) {
        return mx_error(E_FAIL, "triangle is already attached to a cell at the specified index");
    }

    if (tri->partialTriangles[index].neighbors[0] ||
        tri->partialTriangles[index].neighbors[1] ||
        tri->partialTriangles[index].neighbors[2]) {
        return mx_error(E_FAIL, "triangle partial triangles already connected");
    }

    tri->cells[index] = this;

    // index of other cell.
    int otherIndex = index == 0 ? 1 : 0;

    // other cell index could be:
    // null: this is a brand new triangle that's not connected to anything.
    // not null: the other side of the triangle is already connected to another cell.

    // each non-empty cell must have at least one face that's connected to the root cell.
    // but a cell with no triangles has no facets.

    // if the triangle is already attached to another cell, we look for a corresponding
    // facet, if no facet, that means that this is a new fact between this cell, and
    // the other cell the triangle is attached to.

    if(tri->cells[otherIndex] == nullptr) {
        // this creates a facet between root and this cell (if not exists), and
        // adds the tri to it.
        mesh->rootCell()->appendChild(tri, otherIndex);
        assert(tri->cells[otherIndex] == mesh->rootCell());
    }

    std::cout << "tri" << ", {" << tri->vertices[0]->position;
    std::cout << ", " << tri->vertices[1]->position;
    std::cout << ", " << tri->vertices[2]->position;
    std::cout << "}" << std::endl;

    for(int i = 0; i < boundary.size(); ++i) {
        auto pt = boundary[i];
        std::cout << i << ", {" << pt->triangle->vertices[0]->position;
        std::cout << ", " << pt->triangle->vertices[1]->position;
        std::cout << ", " << pt->triangle->vertices[2]->position;
        std::cout << "}" << std::endl;
    }

    if(this == mesh->rootCell()) {
        return S_OK;
    }

    // scan through the list of partial triangles, and connect whichever ones share
    // an edge with the given triangle.
    for(MxPartialTriangle *pt : boundary) {
        for(int k = 0; k < 3; ++k) {
            if(adjacent_triangle_vertices(pt->triangle, tri)) {
                connect_partial_triangles(pt, &tri->partialTriangles[index]);
                assert(adjacent(pt, &tri->partialTriangles[index]));
                break;
            }
        }
    }

    boundary.push_back(&tri->partialTriangles[index]);

    return S_OK;
}

HRESULT MxCell::positionsChanged() {
    area = 0;
    volume = 0;
    centroid = Vector3{0., 0., 0.};
    int ntri = 0;

    for(auto pt : boundary) {
        TrianglePtr tri = pt->triangle;

        ntri += 1;
        centroid += tri->centroid;
        area += tri->area;
        float volumeContr = tri->area * Math::dot(tri->cellNormal(this), tri->centroid);


        float a = tri->area * Math::dot(tri->normal, tri->centroid);
        float b = tri->area * Math::dot(-tri->normal, tri->centroid);

        volume += volumeContr;

        //for(int i = 0; i < 3; ++i) {
        //    float gaussian = 0;
        //    float mean = 0;
        //    discreteCurvature(this, tri->vertices[i], &mean, &gaussian);
        //    pt->vertexAttr[i] = mean;
        //}
    }
    volume /= 3.;
    centroid /= (float)ntri;

    return S_OK;
}

bool MxCell::isRoot() const {
    return this == mesh->rootCell();
}

HRESULT MxCell::appendTriangleFromFacet(TrianglePtr tri, int index) {
    if(tri->cells[0] == this || tri->cells[1] == this) {
        return mx_error(E_FAIL, "triangle is already attached to this cell");
    }

    if (index != 0 && index != 1) {
        return mx_error(E_FAIL, "invalid index argument");
    }

    if (tri->cells[index] != nullptr) {
        return mx_error(E_FAIL, "triangle is already attached to a cell at the specified index");
    }

    if (tri->partialTriangles[index].neighbors[0] ||
        tri->partialTriangles[index].neighbors[1] ||
        tri->partialTriangles[index].neighbors[2]) {
        return mx_error(E_FAIL, "triangle partial triangles already connected");
    }

    tri->cells[index] = this;

    std::cout << "tri" << ", {" << tri->vertices[0]->position;
    std::cout << ", " << tri->vertices[1]->position;
    std::cout << ", " << tri->vertices[2]->position;
    std::cout << "}" << std::endl;

    for(int i = 0; i < boundary.size(); ++i) {
        auto pt = boundary[i];
        std::cout << i << ", {" << pt->triangle->vertices[0]->position;
        std::cout << ", " << pt->triangle->vertices[1]->position;
        std::cout << ", " << pt->triangle->vertices[2]->position;
        std::cout << "}" << std::endl;
    }

    if(this == mesh->rootCell()) {
        return S_OK;
    }

    // scan through the list of partial triangles, and connect whichever ones share
    // an edge with the given triangle.
    for(MxPartialTriangle *pt : boundary) {
        for(int k = 0; k < 3; ++k) {
            if(adjacent_triangle_vertices(pt->triangle, tri)) {
                connect_partial_triangles(pt, &tri->partialTriangles[index]);
                assert(adjacent(pt, &tri->partialTriangles[index]));
                break;
            }
        }
    }

    boundary.push_back(&tri->partialTriangles[index]);

    return S_OK;
}

HRESULT MxCell::topologyChanged() {
    if(renderer) {
        renderer->invalidate();
    }
    return S_OK;
}

HRESULT MxCell::removeTriangleFromFacet(TrianglePtr tri, int index) {

    if (index != 0 && index != 1) {
        return mx_error(E_FAIL, "invalid index argument");
    }

    if(tri->cells[index] != this) {
        return mx_error(E_FAIL, "triangle is not attached to this cell");
    }

    tri->cells[index] = nullptr;

    if(this == mesh->rootCell()) {
        return S_OK;
    }

    // scan through the list of partial triangles, and connect whichever ones share
    // an edge with the given triangle.
    for(MxPartialTriangle *pt : boundary) {
        for(int k = 0; k < 3; ++k) {
            if(adjacent_triangle_vertices(pt->triangle, tri)) {
                disconnect_partial_triangles(pt, &tri->partialTriangles[index]);
                assert(!adjacent(pt, &tri->partialTriangles[index]));
                break;
            }
        }
    }

    remove(boundary, &tri->partialTriangles[index]);

    return S_OK;
}
