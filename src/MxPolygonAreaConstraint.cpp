/*
 * MxPolygonAreaConstraint.cpp
 *
 *  Created on: Sep 20, 2018
 *      Author: andy
 */

#include <MxPolygonAreaConstraint.h>
#include "MxPolygon.h"
#include "MxMesh.h"

MxPolygonAreaConstraint::MxPolygonAreaConstraint(float _targetArea, float _lambda) :
    targetArea{_targetArea}, lambda{_lambda}
{
}

HRESULT MxPolygonAreaConstraint::setTime(float time)
{
    return S_OK;
}

float MxPolygonAreaConstraint::energy(const MxObject** objs, int32_t len)
{
    float e = 0;
    for(int i = 0; i < len; ++i) {
        e += energy(objs[i]);
    }
    return e;
}

HRESULT MxPolygonAreaConstraint::project(MxObject** obj, int32_t len)
{

    for(int i = 0; i < len; ++i) {

        MxPolygon *poly = static_cast<MxPolygon*>(obj[i]);

        float k = energy(poly);

        for(uint i = 0; i < poly->vertices.size(); ++i) {
            VertexPtr vi = poly->vertices[i];
            Vector3 vec = vi->position - poly->centroid;
            vi->position -= k * vec.normalized();
        }

        MeshPtr mesh = poly->cells[0]->mesh;
        mesh->setPositions(0, nullptr);

        float final = energy(poly);

        std::cout << "polygon " << poly->id << " area constraint before/after: " <<
                       k << "/" << final << std::endl;
    }

    return S_OK;
}

float MxPolygonAreaConstraint::energy(const MxObject* obj)
{
    const MxPolygon *poly = static_cast<const MxPolygon*>(obj);
    return lambda * (poly->area - targetArea);
}
