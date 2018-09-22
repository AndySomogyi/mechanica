/*
 * MxPolygonAreaConstraint.cpp
 *
 *  Created on: Sep 20, 2018
 *      Author: andy
 */

#include <MxPolygonAreaConstraint.h>

float MxPolygonAreaConstraint::energy(const MxObject* obj)
{
    return 0;
}

HRESULT MxPolygonAreaConstraint::project(MxObject* obj)
{
    return S_OK;
}
