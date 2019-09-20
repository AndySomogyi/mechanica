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

static MxCellType cellType{"MxCell", MxObject_Type};
MxType *MxCell_Type = &cellType;

bool operator == (const std::array<MxVertex *, 3>& a, const std::array<MxVertex *, 3>& b) {
  return a[0] == b[0] && a[1] == b[1] && a[2] == b[2];
}


bool MxCell::manifold() const {


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

    for(CPPolygonPtr pp : surface) {
        CPolygonPtr poly = pp->polygon;
        MxPolygonType *polyType = static_cast<MxPolygonType*>(poly->ob_type);

        for(int i = 0; i < poly->vertices.size(); ++i) {
            CVertexPtr p1 = poly->vertices[i];
            CVertexPtr p2 = poly->vertices[(i+1)%poly->vertices.size()];

            Color4 edgeColor;


            if (poly == mesh->selectedObject() || poly->edges[i] == mesh->selectedObject()) {
                edgeColor = type->selectedEdgeColor;
            }
            //else if(Mx_IsEdgeToTriangleConfiguration(poly->edges[i])) {
            //    edgeColor = Color4::red();
            //}
            else {
                edgeColor = polyType->edgeColor;
            }

            attrs[0].position = p1->position;
            attrs[0].color = edgeColor;
            attrs[1].position = p2->position;
            attrs[1].color = edgeColor;
            attrs[2].position = poly->centroid;
            attrs[2].color = polyType->centerColor;

            ptr += 3 * stride;
            attrs = (VertexAttribute*)ptr;
        }
    }

    assert(ptr == vertexCount * stride + (uchar*)buffer);
}









void MxCell::dump() const {

    std::cout << "Cell { name: \"" << name << "\", id: " << id
              << ", area: " << area << ", volume: " << volume
              << ", centroid: " << centroid  << "}" << std::endl;
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
    out << surface.size()  << std::endl;
    for (int i = 0; i < surface.size(); ++i) {
        const MxPolygon &face = *surface[i]->polygon;
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

HRESULT MxCell::positionsChanged() {
    area = 0;
    volume = 0;
    centroid = Vector3{0., 0., 0.};
    int npoly = 0;

    for(auto pt : surface) {
        PolygonPtr poly = pt->polygon;

        assert(poly->area >= 0);

        npoly += 1;
        centroid += poly->centroid;
        area += poly->area;
        volume += poly->volume(this);
    }

    centroid /= (float)npoly;
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
    return true;
}

Vector3 MxCell::centerOfMass() const
{
    std::set<VertexPtr> verts;
    Vector3 sum;
    float mass = 0;


    for(PPolygonPtr pt : surface) {
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

    for(PPolygonPtr pt : surface) {
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

    for(PPolygonPtr pt : surface) {
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


uint MxCell::faceCount()
{
    uint fc = 0;
    for(CPPolygonPtr poly : surface ) {
        fc += poly->polygon->vertices.size();
    }
    return fc;
}

uint MxCell::vertexCount()
{
    uint vc = 0;
    for(CPPolygonPtr poly : surface ) {
        vc += 3 * poly->polygon->vertices.size();
    }
    return vc;
}

std::ostream& operator <<(std::ostream &stream, const MxCell *cell)
{
    stream << "Cell { name: \"" << cell->name << "\", id: " << cell->id
              << ", area: " << cell->area << ", volume: " << cell->volume
              << ", centroid: " << cell->centroid  << "}" << std::endl;

    return stream;
}
