/*
 * MxCellVolumeConstraint.cpp
 *
 *  Created on: Sep 20, 2018
 *      Author: andy
 */

#include <MxCellVolumeConstraint.h>
#include "MxMesh.h"

float MxCellVolumeConstraint::energy(const CObject **objs, int32_t len)
{
    float e = 0;
    for(int i = 0; i < len; ++i) {
        e += energy(objs[i]);
    }
    return e;
}

HRESULT MxCellVolumeConstraint::project(CObject **objs, int32_t len)
{
    for(int i = 0; i < len; ++i) {
        MxCell *cell = static_cast<MxCell*>(objs[i]);
        float vc = energy(cell);

        for(PPolygonPtr pp : cell->surface) {

            PolygonPtr poly = pp->polygon;

            for(int j = 0; j < poly->vertices.size(); ++j) {
                VertexPtr v = poly->vertices[j];
                v->position -= vc * (1/3.) * poly->vertexNormal(j, cell);
                checkVec(v->position);
            }
        }

        MeshPtr mesh = cell->mesh;
        mesh->setPositions(0, nullptr);


        float final = energy(cell);

        std::cout << "cell " << cell->id << " volume constraint before/after: " <<
                vc << "/" << final << std::endl;

    }
    return S_OK;
}

MxCellVolumeConstraint::MxCellVolumeConstraint(float _targetVolume, float _lambda) :
    targetVolume{_targetVolume}, lambda{_lambda}
{
}

HRESULT MxCellVolumeConstraint::setTime(float time)
{
    return S_OK;
}

float MxCellVolumeConstraint::energy(const CObject* obj)
{
    const MxCell *cell = static_cast<const MxCell*>(obj);
    return lambda * (cell->volume - targetVolume);
}
