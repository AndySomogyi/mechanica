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
#include <set>

#include "MxDebug.h"
#include "DifferentialGeometry.h"
#include "MxMeshRenderer.h"

bool operator == (const std::array<MxVertex *, 3>& a, const std::array<MxVertex *, 3>& b) {
  return a[0] == b[0] && a[1] == b[1] && a[2] == b[2];
}


bool MxCell::manifold() const {

    for(auto t : boundary) {

        // check pointers
        if (!adjacentPartialTrianglePointers(t, t->neighbors[0]) ||
            !adjacentPartialTrianglePointers(t, t->neighbors[1]) ||
            !adjacentPartialTrianglePointers(t, t->neighbors[2])) {
            return false;
        }

        // check vertices
        if (!adjacentTriangleVertices(t->triangle, t->neighbors[0]->triangle) ||
            !adjacentTriangleVertices(t->triangle, t->neighbors[1]->triangle) ||
            !adjacentTriangleVertices(t->triangle, t->neighbors[2]->triangle)) {
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

    connectTriangleCell(tri, nullptr, index);
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

    connectTriangleCell(pt->triangle, this, ptIndx);
    
    /*
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
     */

    // scan through the list of partial triangles, and connect whichever ones share
    // an edge with the given triangle.
    for(PTrianglePtr t : boundary) {
        for(int k = 0; k < 3; ++k) {
            if(adjacentTriangleVertices(pt->triangle, t->triangle)) {
                connectPartialTrianglePartialTriangle(pt, t);
                assert(adjacentPartialTrianglePointers(pt, t));
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
            if(pt->triangle->orientation() != Orientation::Outward) {
                pt->triangle->color = Color4{0., 1., 0., 0.3};
            }
            else if (pt->triangle->color == Color4{0., 1., 0., 0.3} ) {
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

Vector3 MxCell::centerOfMass() const
{
    std::set<VertexPtr> verts;
    Vector3 sum;
    float mass = 0;


    for(PTrianglePtr pt : boundary) {
        for(VertexPtr v : pt->triangle->vertices) {
            if(verts.find(v) == verts.end()) {
                mass += v->mass;
                sum += v->mass * v->position;
                verts.insert(v);
            }
        }
    }

    return sum / mass;
}

Vector3 MxCell::radiusMeanVarianceStdDev() const
{
    int npts = 0;
    float radius = 0;
    float variance = 0;

    std::set<VertexPtr> verts;
    //  sqrt((Xp-Xc)^2 + (Yp-Yc)^2 + (Zp-Zc)^2) - R

    for(PTrianglePtr pt : boundary) {
        for(VertexPtr v : pt->triangle->vertices) {
            if(verts.find(v) == verts.end()) {
                float x = centroid[0] - v->position[0];
                float y = centroid[1] - v->position[1];
                float z = centroid[2] - v->position[2];
                radius += sqrt(x*x + y*y + z*z);
                npts += 1;
                verts.insert(v);
            }
        }
    }

    radius = radius / npts;

    for(VertexPtr v : verts) {
        float x = centroid[0] - v->position[0];
        float y = centroid[1] - v->position[1];
        float z = centroid[2] - v->position[2];
        float xMu = radius - sqrt(x*x + y*y + z*z);
        variance += xMu * xMu;
    }

    variance = variance / npts;

    return {{radius, variance, sqrt(variance)}};
}

Matrix3 MxCell::momentOfInertia() const
{
    Matrix3 inertia;

    std::set<VertexPtr> verts;
    //  sqrt((Xp-Xc)^2 + (Yp-Yc)^2 + (Zp-Zc)^2) - R

    for(PTrianglePtr pt : boundary) {
        for(VertexPtr v : pt->triangle->vertices) {
            if(verts.find(v) == verts.end()) {

                inertia[0][0]  += v->mass * (centroid[0] * centroid[0]  - v->position[0] * v->position[0]);
                inertia[0][1]  += v->mass * v->position[0] * v->position[1];
                inertia[0][2]  += v->mass * v->position[0] * v->position[2];

                inertia[1][0]  += v->mass * v->position[0] * v->position[1];
                inertia[1][1]  += v->mass * (centroid[1] * centroid[1]  - v->position[1] * v->position[1]);
                inertia[1][2]  += v->mass * v->position[1] * v->position[2];

                inertia[2][0]  += v->mass * v->position[0] * v->position[2];
                inertia[2][1]  += v->mass * v->position[1] * v->position[2];
                inertia[2][2]  += v->mass * (centroid[2] * centroid[2]  - v->position[2] * v->position[2]);

                verts.insert(v);
            }
        }
    }

    return inertia;
}

float MxCell::volumeConstraint()
{
    return 0.05 * (volume - targetVolume);
}

void MxCell::projectVolumeConstraint()
{

    float c = volumeConstraint();

    float sumWC = 0;

    std::set<VertexPtr> verts;

    for(PTrianglePtr pt : boundary) {
        TrianglePtr tri = pt->triangle;

        for(VertexPtr v : tri->vertices) {
            if(verts.find(v) == verts.end()) {
                verts.insert(v);
                float w = (1. / v->mass);
                v->awc = v->areaWeightedNormal(this);
                sumWC += w * Math::dot(v->awc, v->awc);
            }
        }
    }

    for(VertexPtr v : verts) {
        v->position -= (1. / v->mass * c) / (sumWC) * v->awc;
    }

    //for(VertexPtr v : verts) {
    //    for(TrianglePtr tri : v->_triangles) {
    //        tri->positionsChanged();
    //    }
    //}

    //for(PTrianglePtr pt : boundary) {
    //    TrianglePtr tri = pt->triangle;
    //    tri->positionsChanged();
    //}

    /*
    std::set<TrianglePtr> tris;
    for(VertexPtr v : verts) {
        for(TrianglePtr tri : v->_triangles) {
            if(tris.find(tri) == tris.end()) {
                tris.insert(tri);
                tri->positionsChanged();
            }
        }
    }
    */

}
