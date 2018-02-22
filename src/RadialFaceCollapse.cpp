/*
 * RadialFaceCollapse.cpp
 *
 *  Created on: Feb 20, 2018
 *      Author: andy
 */


#include <RadialFaceCollapse.h>
#include "MxMesh.h"
#include <cmath>

using namespace Magnum;



static float angleBetweenFaces(PTrianglePtr a, PTrianglePtr b);


RadialFaceCollapse::~RadialFaceCollapse()
{
    // TODO Auto-generated destructor stub
}

HRESULT RadialFaceCollapse::apply()
{
}

float RadialFaceCollapse::energy() const
{
}

bool RadialFaceCollapse::depends(CTrianglePtr) const
{
}

bool RadialFaceCollapse::depends(CVertexPtr) const
{
}

bool RadialFaceCollapse::equals(const Edge& e) const
{
}

bool RadialFaceCollapse::equals(CVertexPtr) const
{
}


MeshOperation* RadialFaceCollapse::create(MeshPtr mesh, TrianglePtr tri)
{
    float angle = std::numeric_limits<float>::max();
    PTrianglePtr a = nullptr;
    PTrianglePtr b = nullptr;
    CellPtr cell;

    for(int i = 0; i < 2; ++i) {
        for(int j = 0; j < 3; ++j) {
            float ang = angleBetweenFaces(&tri->partialTriangles[i], tri->partialTriangles[i].neighbors[j]);
            if(ang < angle ) {
                angle = ang;
                a = &tri->partialTriangles[i];
                b = tri->partialTriangles[i].neighbors[j];
                cell = tri->cells[i];
            }
        }
    }

    if (angle <= mesh->getMinAngleCutoff()) {
        return new RadialFaceCollapse(mesh, cell, a, b);
    }

    return nullptr;
}

void RadialFaceCollapse::mark() const
{
}

RadialFaceCollapse::RadialFaceCollapse(MeshPtr m,
        CellPtr cell, PTrianglePtr a, PTrianglePtr b) :
                MeshOperation(m)
{
}


static float angleBetweenFaces(PTrianglePtr a, PTrianglePtr b) {

    int a_indx = &a->triangle->partialTriangles[0] == a ? 0 : 1;
    Vector3 a_normal = a->triangle->normal * (a_indx == 0 ? 1 : -1);
    int b_indx = &b->triangle->partialTriangles[0] == b ? 0 : 1;
    Vector3 b_normal = b->triangle->normal * (b_indx == 0 ? 1 : -1);

    assert(a->triangle->cells[a_indx] == b->triangle->cells[b_indx]);

    float dot = Math::dot(a_normal, b_normal);

    return std::fmod(std::acos(dot) + M_PI, M_PI);
}
