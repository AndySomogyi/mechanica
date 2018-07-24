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
        if(connectedTriangleCellPointers(tri, cell)) {
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
                if(incidentVertexTriangle(v, next)) {
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
            if(!incidentVertexTriangle(v, prev)) {
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
                force = tri->force(i);
                break;
            }
        }
        forceSum += Math::dot(force, (tri->centroid - v->position).normalized());
        //areaSum += tri->area;
    }
    return forceSum;
}

static float max3(float a, float b, float c )
{
   float max = ( a < b ) ? b : a;
   return ( ( max < c ) ? c : max );
}



// Constructs a plane from a collection of points
// so that the summed squared distance to all points is minimzized
HRESULT planeFromPoints(const std::vector<CVertexPtr> &pts, Vector3 &normal, Vector3 &point)  {

    int n = pts.size();
    if(n < 3) {
        return mx_error(E_FAIL, "At least three points required");
    }


    Vector3 sum = Vector3{0.0f, 0.0f, 0.0f};
    for(CVertexPtr vert : pts) {
        sum += vert->position;
    }

    Vector3 centroid = sum * (1.0 / n);

    // Calc full 3x3 covariance matrix, excluding symmetries:
    float xx = 0.0, xy = 0.0, xz = 0.0;
    float yy = 0.0, yz = 0.0, zz = 0.0;

    for(CVertexPtr vert : pts) {
        Vector3 p = vert->position;
        Vector3 r = p - centroid;
        xx += r.x() * r.x();
        xy += r.x() * r.y();
        xz += r.x() * r.z();
        yy += r.y() * r.y();
        yz += r.y() * r.z();
        zz += r.z() * r.z();
    }

    float det_x = yy*zz - yz*yz;
    float det_y = xx*zz - xz*xz;
    float det_z = xx*yy - xy*xy;

    float det_max = max3(det_x, det_y, det_z);

    if(det_max <= 0.) {
        return mx_error(E_FAIL,  "The points don't span a plane");
    }

    // Pick path with best conditioning:
    Vector3 dir;
    if (det_max == det_x) {
        dir = Vector3{
            det_x,
            xz*yz - xy*zz,
            xy*yz - xz*yy,
        };
    }
    else if (det_max == det_y) {
        dir = Vector3{
            xz*yz - xy*zz,
            det_y,
            xy*xz - yz*xx,
        };
    } else {
        dir = Vector3{
            xy*yz - xz*yy,
            xy*xz - yz*xx,
            det_z,
        };
    };

    normal = dir.normalized();
    point = centroid;

    return S_OK;
}

Vector3 centroid(const std::vector<CVertexPtr>& pts)
{
    Vector3 sum = Vector3{0.0f, 0.0f, 0.0f};
    for(CVertexPtr vert : pts) {
        sum += vert->position;
    }

    return sum * (1.0 / pts.size());
}

float forceDivergenceForCell(CVertexPtr v, CCellPtr c)
{
    Vector3 force;
    float forceSum = 0;
    float areaSum = 0;

    // get the first triangle
    TrianglePtr first = v->triangleForCell(c);
    // the loop triangle
    TrianglePtr tri = first;
    // keep track of the previous triangle
    TrianglePtr prev = nullptr;

    do {
        for(int i = 0; i < 3; ++i) {
            if(tri->vertices[i] == v) {
                force = tri->partialTriangles[tri->cellIndex(c)].force[i];

                forceSum += Math::dot(force, (tri->centroid - v->position));
                break;
            }
        }

        TrianglePtr next = tri->nextTriangleInFan(v, c, prev);
        prev = tri;
        tri = next;
    } while(tri && tri != first);

    return forceSum;
}

Vector3 centroidTriangleFan(CVertexPtr center, const std::vector<TrianglePtr>& tri)
{
    assert(tri.size() >= 3);
    CVertexPtr prev = nullptr;
    for(int j = 0; j < 3; ++j) {
        if(tri[0]->vertices[j] != center && incidentVertexTriangle(tri[0]->vertices[j], tri[tri.size()-1])) {
            prev = tri[0]->vertices[j];
            break;
        }
    }
    assert(prev);

    Vector3 sum = Vector3{0.0f, 0.0f, 0.0f};

    sum += prev->position;

    for(int i = 1; i < tri.size(); ++i) {
        CTrianglePtr t = tri[i];
        for(int j = 0; j < 3; ++j) {
            if(t->vertices[j] != center && t->vertices[j] != prev) {
                prev = t->vertices[j];
                sum += prev->position;
                break;
            }
        }

    }

    return sum * (1.0 / tri.size());
}

float gaussianCurvature(CVertexPtr vert, CCellPtr cell)
{
    Vector3 a, b;

    float thetaSum = 0;
    float areaSum = 0;

    // get the first triangle
    TrianglePtr first = vert->triangleForCell(cell);
    // the loop triangle
    TrianglePtr tri = first;
    // keep track of the previous triangle
    TrianglePtr prev = nullptr;
    do {


        // vectors from center to two outside vertices
        for(int i = 0; i < 3; ++i) {
            if(tri->vertices[i] == vert) {
                a = tri->vertices[(i+1)%3]->position - vert->position;
                b = tri->vertices[(i+2)%3]->position - vert->position;
                break;
            }
        }

        assert(tri->area  > 0);

        float cosTheta = Math::dot(a, b) / (a.length() * b.length());
        float theta = acos(cosTheta);
        thetaSum += theta;
        areaSum += tri->area;


        TrianglePtr next = tri->nextTriangleInFan(vert, cell, prev);
        prev = tri;
        tri = next;
    } while(tri && tri != first);

    assert(thetaSum >= 0. && thetaSum <= 2. * M_PI);

    return (2. * M_PI - thetaSum) / areaSum;
}

float umbrella(CVertexPtr vert, CCellPtr cell) {
    Vector3 sum;

    float eSum = 0;
    
    // get the first triangle
    TrianglePtr first = vert->triangleForCell(cell);
    // the loop triangle
    TrianglePtr tri = first;
    // keep track of the previous triangle
    TrianglePtr prev = nullptr;
    do {

        Vector3 diff = vert->position - tri->centroid;
        float e = diff.length();
        sum += diff / e;
        eSum += e;
        
        
        TrianglePtr next = tri->nextTriangleInFan(vert, cell, prev);
        prev = tri;
        tri = next;
    } while(tri && tri != first);
    
    return 2. / eSum * sum.length();
    

}

Vector3 normalTriangleFan(CCellPtr cell, const std::vector<TrianglePtr>& triFan)
{
    Vector3 sum;

    for(const TrianglePtr tri : triFan) {
        sum += tri->cellNormal(cell) * tri->area;
    }

    return sum.normalized();
}

Vector3 centroid(const std::vector<Vector3> &pts) {
    Vector3 sum;
    for(const Vector3 &vec : pts) {
        sum += vec;
    }
    return sum / pts.size();
}

