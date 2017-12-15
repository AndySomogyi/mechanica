/*
 * DifferentialGeometry.cpp
 *
 *  Created on: Dec 8, 2017
 *      Author: andy
 */
#define _USE_MATH_DEFINES
#include <cmath>
#include "DifferentialGeometry.h"
#include <iostream>

using namespace Magnum;

HRESULT discreteCurvature(CCellPtr cell, CVertexPtr vert, float* meanCurvature,
        float* gaussianCurvature) {
    for(CTrianglePtr tri : vert->triangles()) {
        if(incident(tri, cell)) {
            return discreteCurvature(cell, vert, tri, meanCurvature, gaussianCurvature);
        }
    }
    return mx_error(E_FAIL, "vertex is not incident to cell");
}

HRESULT discreteCurvature(CCellPtr cell, CVertexPtr vi, CTrianglePtr tri,
        float* meanCurvature, float* gaussianCurvature) {

    const CTrianglePtr first = tri;

    Vector3 ksum;

    // vertex normal sum
    Vector3 n_sum;

    float areaSum = 0;

    float thetaSum = 0;

    // keep track of the previous triangle
    CTrianglePtr prev = nullptr;
    do {
        CTrianglePtr next = tri->nextTriangleInFan(vi, cell, prev);
        prev = tri;
        tri = next;

        assert(prev && next);

        CVertexPtr v_alpha = nullptr;
        CVertexPtr v_beta = nullptr;
        CVertexPtr vj = nullptr;

        for(int i = 0; i < 3; ++i) {
            CVertexPtr v = prev->vertices[i];
            if(v != vi) {
                if(incident(v, next)) {
                    assert(!vj);
                    vj = v;
                }
                else {
                    assert(!v_alpha);
                    v_alpha = v;
                }
            }
        }

        for(int i = 0; i < 3; ++i) {
            CVertexPtr v = next->vertices[i];
            if(!incident(v, prev)) {
                v_beta = v;
                break;
            }
        }

        assert(v_alpha && v_beta && vj);

        Vector3 x_alpha_i = vi->position - v_alpha->position;
        Vector3 x_alpha_j = vj->position - v_alpha->position;
        Vector3 x_beta_i = vi->position - v_beta->position;
        Vector3 x_beta_j = vj->position - v_beta->position;
        Vector3 x_ij = vi->position - vj->position;

        float alpha_i_len = x_alpha_i.length();
        float alpha_j_len = x_alpha_j.length();
        float beta_i_len = x_beta_i.length();
        float beta_j_len = x_beta_j.length();
        float x_ij_len = x_ij.length();

        float cos_alpha = Math::dot(x_alpha_i, x_alpha_j) / (alpha_i_len * alpha_j_len);
        float cos_beta = Math::dot(x_beta_i, x_beta_j) / (beta_i_len * beta_j_len);
        float sin_alpha = ::sin(::acos(cos_alpha));
        float sin_beta = ::sin(::acos(cos_alpha));
        float theta = ::acos(Math::dot(x_alpha_i, x_ij) / (alpha_i_len * x_ij_len));

        ksum += (cos_alpha / sin_alpha + cos_beta / sin_beta) * x_ij;
        thetaSum += theta;
        // use total triangle area for now
        areaSum += prev->area;

        // cross product is triangle surface normal * triangle area.
        // area-weighted sum of triangle normals. similar to angle weighted sum
        n_sum += Math::cross(-x_alpha_i, x_ij);

    } while(tri && tri != first);

    *gaussianCurvature = (2. * M_PI - thetaSum) / ((1./3) * areaSum);
    *meanCurvature = (1./4) * Math::dot(ksum, n_sum.normalized());

    return S_OK;
}

float forceDivergence(CVertexPtr v) {
    Vector3 force;
    float forceSum = 0;
    float areaSum = 0;
    for(CTrianglePtr tri : v->triangles()) {
        for(int i = 0; i < 3; ++i) {
            if(tri->vertices[i] == v) {
                force = tri->force[i];
                break;
            }
        }
        forceSum += Math::dot(force, (tri->centroid - v->position).normalized());
        //areaSum += tri->area;
    }
    return forceSum;
}
