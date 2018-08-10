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
        if (!adjacentTriangleVertices(t->polygon, t->neighbors[0]->polygon) ||
            !adjacentTriangleVertices(t->polygon, t->neighbors[1]->polygon) ||
            !adjacentTriangleVertices(t->polygon, t->neighbors[2]->polygon)) {
            return false;
        }
    }
    return true;
}

void MxCell::vertexAtributeData(const std::vector<MxVertexAttribute>& attributes,
        uint vertexCount, uint stride, void* buffer) {
    MxCellType *type = (MxCellType*)ob_type;


    uchar *ptr = (uchar*)buffer;

    // vertex attribute, currently we just dump out triangles,
    // so each subsequent vertex attr represent a vertex, and three
    // Contiguous attrs represent a triangle.
    VertexAttribute *attrs = (VertexAttribute*)ptr;

    // for each triangle,
    // ptr += 3 * stride
    // VertexAttribute *attrs = (VertexAttribute*)ptr;

    // counter of current triangle being written to buffer
    uint triCount = 0;

    for(CPPolygonPtr pp : boundary) {
        CPolygonPtr poly = pp->polygon;

        for(int i = 0; i < poly->vertices.size(); ++i) {
            CVertexPtr p1 = poly->vertices[i];
            CVertexPtr p2 = poly->vertices[(i+1)%poly->vertices.size()];



            attrs[0].position = p1->position;
            attrs[0].color = Color4::green();
            attrs[1].position = p2->position;
            attrs[1].color = Color4::green();
            attrs[2].position = poly->centroid;
            attrs[2].color = Color4::red();

            ptr += 3 * stride;
            attrs = (VertexAttribute*)ptr;
        }
    }

    assert(ptr == vertexCount * stride + (uchar*)buffer);
}









void MxCell::dump() {

    for (uint i = 0; i < boundary.size(); ++i) {
        const MxPolygon &ti = *boundary[i]->polygon;

        std::cout << "face[" << i << "] {" << std::endl;
        //std::cout << "vertices:" << ti.vertices << std::endl;
        std::cout << "}" << std::endl;
    }

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
        const MxPolygon &face = *boundary[i]->polygon;
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

HRESULT MxCell::updateDerivedAttributes() {
    area = 0;
    volume = 0;
    centroid = Vector3{0., 0., 0.};
    int ntri = 0;

    for(auto pt : boundary) {
        PolygonPtr tri = pt->polygon;

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
        for(PPolygonPtr pt : boundary) {
            if(pt->polygon->orientation() != Orientation::Outward) {
                pt->polygon->color = Color4{0., 1., 0., 0.3};
            }
            else if (pt->polygon->color == Color4{0., 1., 0., 0.3} ) {
                pt->polygon->color = Color4{0., 0., 0., 0.};
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
            if(!boundary[i]->polygon->isValid()) {
                badCells.push_back(boundary[i]->polygon->id);
            }

            std::cout << boundary[i]->polygon << std::endl;
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


    for(PPolygonPtr pt : boundary) {
        for(VertexPtr v : pt->polygon->vertices) {
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

    for(PPolygonPtr pt : boundary) {
        for(VertexPtr v : pt->polygon->vertices) {
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

    for(PPolygonPtr pt : boundary) {
        for(VertexPtr v : pt->polygon->vertices) {
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

    for(PPolygonPtr pt : boundary) {
        PolygonPtr tri = pt->polygon;

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

    //for(PPolygonPtr pt : boundary) {
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

uint MxCell::faceCount()
{
    uint fc = 0;
    for(CPPolygonPtr poly : boundary ) {
        fc += poly->polygon->vertices.size();
    }
    return fc;
}

uint MxCell::vertexCount()
{
    uint vc = 0;
    for(CPPolygonPtr poly : boundary ) {
        vc += 3 * poly->polygon->vertices.size();
    }
    return vc;
}
