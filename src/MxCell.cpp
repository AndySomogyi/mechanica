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

        if(false) {
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
                color = type->color(this);
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

    connect_triangle_cell(tri, nullptr, index);
    //tri->cells[index] = nullptr;


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

HRESULT MxCell::appendChild(PTrianglePtr pt) {

    if(!pt) {
        return mx_error(E_FAIL, "partial triangle is null");
    }

    int ptIndx = &pt->triangle->partialTriangles[0] == pt ? 0 : 1;

    if(pt->triangle->cells[ptIndx] == this) {
        return mx_error(E_FAIL, "partial triangle is already attached to this cell");
    }

    if(pt->triangle->cells[(ptIndx+1)%2] == this) {
        return mx_error(E_FAIL, "opposite partial triangle is already attached to this cell");
    }

    if (pt->triangle->cells[ptIndx] != nullptr) {
        return mx_error(E_FAIL, "partial triangle is already attached to a cell at the specified index");
    }

    if (pt->neighbors[0] ||
        pt->neighbors[1] ||
        pt->neighbors[2]) {
        return mx_error(E_FAIL, "partial triangle neighbors already connected");
    }

    connect_triangle_cell(pt->triangle, this, ptIndx);
    //pt->triangle->cells[ptIndx] = this;

    std::cout << __PRETTY_FUNCTION__ << "{"
        << "\triangle:{id:" << pt->triangle->id
        << ", pos:{" << pt->triangle->vertices[0]->position
        << ", " << pt->triangle->vertices[1]->position
        << ", " << pt->triangle->vertices[2]->position
        << "}}}" << std::endl;

    for(int i = 0; i < boundary.size(); ++i) {
        auto t = boundary[i];
        std::cout << i << ", {" << t->triangle->vertices[0]->position;
        std::cout << ", " << t->triangle->vertices[1]->position;
        std::cout << ", " << t->triangle->vertices[2]->position;
        std::cout << "}" << std::endl;
    }

    // scan through the list of partial triangles, and connect whichever ones share
    // an edge with the given triangle.
    for(PTrianglePtr t : boundary) {
        for(int k = 0; k < 3; ++k) {
            if(adjacent_triangle_vertices(pt->triangle, t->triangle)) {
                connect_partial_triangles(pt, t);
                assert(adjacent(pt, t));
                break;
            }
        }
    }

    boundary.push_back(pt);

    return S_OK;
}

HRESULT MxCell::updateDerivedAttributes() {
    area = 0;
    volume = 0;
    centroid = Vector3{0., 0., 0.};
    int ntri = 0;

    for(auto pt : boundary) {
        TrianglePtr tri = pt->triangle;

        assert(tri->area >= 0);

        ntri += 1;
        centroid += tri->centroid;
        area += tri->area;
        float volumeContr = tri->area * Math::dot(tri->cellNormal(this), tri->centroid);
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

    //std::cout << "cell id:" << id << ", volume:" << volume << ", area:" << area << std::endl;

    if(!isRoot()) {
        for(PTrianglePtr pt : boundary) {
            if(Math::dot(pt->triangle->cellNormal(this), pt->triangle->centroid - centroid) <= 0) {
                pt->triangle->color = Color4{0., 1., 0., 0.3};
            }
            else if (pt->triangle->color == Color4{0., 1., 0., 0.3}) {
                pt->triangle->color = Color4{0., 0., 0., 0.};
            }
        }
    }

    return S_OK;
}

bool MxCell::isRoot() const {
    return this == mesh->rootCell();
}

HRESULT MxCell::topologyChanged() {
    if(renderer) {
        renderer->invalidate();
    }
    return S_OK;
}

bool MxCell::isValid() const
{
    bool result = true;

    for(int i = 0; i < boundary.size(); ++i) {
        result &= boundary[i]->isValid();
    }

    if(!result) {
        std::cout << "cell not valid" << std::endl;

        std::vector<uint> badCells;

        for(int i = 0; i < boundary.size(); ++i) {
            if(!boundary[i]->triangle->isValid()) {
                badCells.push_back(boundary[i]->triangle->id);
            }

            std::cout << boundary[i]->triangle << std::endl;
        }

        std::cout << "bad cells:";
        for(auto i : badCells) {
            std::cout << i << ",";
        }

        return false;

    }

    if(!manifold()) {
        std::cout << "error, cellId:" << id << " is not manifold" << std::endl;
        return false;
    }

    return true;
}


