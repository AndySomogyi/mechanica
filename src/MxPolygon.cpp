/*
 * MxTriangle.cpp
 *
 *  Created on: Oct 3, 2017
 *      Author: andy
 */

#include <MxPolygon.h>
#include "MxCell.h"
#include "MxDebug.h"
#include <iostream>

static std::string to_string(CCellPtr cell) {
    return cell ? std::to_string(cell->id) : "null";
}

static std::string to_string(const MxPartialPolygon *pt) {
    return "triId:" + (pt ? std::to_string(pt->polygon->id) : "null");
}

std::ostream& operator<<(std::ostream& os, CPolygonPtr tri)
{
    os << "Polygon {" << std::endl
       << "id:" << tri->id << "," << std::endl
       << "cells:{" << to_string(tri->cells[0]) << "," << to_string(tri->cells[1]) << "}," << std::endl
       << "vertices:{" << std::endl
       << "\t" << tri->vertices[0] << ", " << std::endl
       << "\t" << tri->vertices[1] << ", " << std::endl
       << "\t" << tri->vertices[2] << "}" << std::endl

       << "}" << std::endl;
    return os;
}


MxPolygon::MxPolygon(uint _id, MxPolygonType* type,
        const std::vector<VertexPtr>& verts,
        const std::array<CellPtr, 2>& cells,
        const std::array<MxPartialPolygonType*, 2>& partTriTypes) :
            id{_id}, MxObject{type}, vertices{verts}, cells{cells},
            partialTriangles{{{partTriTypes[0], this}, {partTriTypes[1], this}}} {


    neighbors.resize(vertices.size());
    partialTriangles[0].force.resize(vertices.size());
    partialTriangles[1].force.resize(vertices.size());
    _vertexNormals.resize(vertices.size());
                _vertexAreas.resize(vertices.size());

    positionsChanged();
}

/**
 * Neighbor triangle indexes are related to vertex indexes as
 * the i'th neighbor triangle shares vertices at indexes i and either i+1
 * or i-1.
 */
int MxPolygon::adjacentEdgeIndex(CVertexPtr a, CVertexPtr b) const {
    for(int i = 0; i < 3; ++i) {
        if((vertices[i] == a && vertices[(i+1)%3] == b) ||
           (vertices[i] == b && vertices[(i+1)%3] == a)) {
            return i;
        }
    }
    return -1;
}

HRESULT MxPolygon::positionsChanged() {

    area = 0.;
    normal = {{0.f, 0.f, 0.f}};
    _volume = 0.f;
    centroid = {{0.f, 0.f, 0.f}};

    for (CVertexPtr v : vertices) {
        centroid += v->position;
    }

    centroid /= (float)vertices.size();


    // triangle area is 1/2 * length of surface normal (non-normalized)


    for (int i = 0; i < vertices.size(); ++i) {
        int prevIndex = (vertices.size() + ((i-1)%vertices.size())) % vertices.size();
        int nextIndex = (i+1)%vertices.size();

        CVertexPtr vp = vertices[prevIndex];
        CVertexPtr v = vertices[i];
        CVertexPtr vn = vertices[nextIndex];

        Vector3 np = triangleNormal(v->position, (v->position + vp->position) / 2., centroid);
        Vector3 nn = triangleNormal((vn->position + v->position) / 2., v->position, centroid);
        Vector3 vertNormal = (np + nn);
        float vertLen = vertNormal.length();

        // normalized surface normals and vertex areas
        _vertexNormals[i] = vertNormal / vertLen;
        _vertexAreas[i] = 0.5 * vertLen;

        Vector3 triCentroid = (v->position + vn->position + centroid) / 3.;
        Vector3 triNormal = triangleNormal(v->position, vn->position, centroid);

        // Volume contribution for each triangle is
        // (A/3) * ( N_x*(x1 + x2 + x3) + N_y * (y1 + y2 + y3) + N_z * (z1 + z2 + x3)
        // where A is the triangle area, and N is the normalized surface normal vector.
        // A = |surface normal| / 2
        // note, (1/3) * ( N_x*(x1 + x2 + x3) + N_y * (y1 + y2 + y3) + N_z * (z1 + z2 + x3))
        // is N dot centroid
        // and A = |non-normalized surface normal| / 2


        _volume += 0.5 * Math::dot(triNormal, triCentroid);

        // polygon area
        area += 0.5f * triNormal.length();;

        // non-normalized polygon normal
        normal += triNormal;
    }

    normal = normal.normalized();

    // total volume contribution
    _volume /= 3.;

#ifndef NDEBUG
    float vertArea = 0;
    for (float v :_vertexAreas) {
        vertArea += v;
    }

    std::cout << "vert area: " << vertArea << ", poly area: " << area << ", diff: " << area - vertArea << std::endl;
#endif

    return S_OK;
}


bool MxPolygon::isConnected() const {


    return true;

}

Vector3 MxPolygon::vertexNormal(uint i, CCellPtr cell) const
{
    assert(cells[0] == cell || cells[1] == cell);
    float direction = cells[0] == cell ? 1.0f : -1.0f;

    return direction * _vertexNormals[i];
}

bool MxPolygon::isValid() const  {



    for(int i = 0; i < 3; ++i) {
        VertexPtr v1 = vertices[i];
        VertexPtr v2 = vertices[(i+1)%3];
        int ni = adjacentEdgeIndex(v1, v2);
        assert(ni == i);


    }

    for(int i = 0; i < 2; ++i) {
        if(!cells[i]) {
            std::cout << "error, triangle:" << this << std::endl
                    << ", cell[" << i << "] is null" << std::endl;
            return false;
        }

        if(!connectedPolygonCellPointers(const_cast<PolygonPtr>(this), cells[i])) {
            std::cout << "error, triangle:" << this << std::endl
                    << ", triangle is not incident to cell[" << i << "]" << std::endl;
            return false;
        }

        if(cells[i]->isRoot()) {
            assert(partialTriangles[i].mass == 0.);
        } else {
            isfinite(partialTriangles[i].mass ) && partialTriangles[i].mass  > 0;
        }
    }

    if(!isConnected()) {
        std::cout << "error, triangle:" << this << std::endl
                << " is not connected" << std::endl;
        return false;
    }

    if(!isfinite(area)) {
        std::cout << "error, triangle:" << this << std::endl
                << ", area is not finite" << std::endl;
        return false;
    }

    if(area < 0) {
        std::cout << "error, triangle:" << this << std::endl
                << ", area is negative" << std::endl;
        return false;
    }

    //if(!isfinite(aspectRatio)) {
    //    std::cout << "error, triangle:" << this << std::endl
    //            << ", aspect ratio is not finite" << std::endl;
    //    return false;
    //}

    //if(aspectRatio <= 0) {
    //    std::cout << "error, triangle:" << this << std::endl
    //            << ", aspect ratio is negative" << std::endl;
    //    return false;
    //}

    if(!isfinite(getMass())) {
        std::cout << "error, triangle:" << this << std::endl
                << ", mass is not finite" << std::endl;
        return false;
    }

    if(getMass() < 0) {
        std::cout << "error, triangle:" << this << std::endl
                << ", mass is negative" << std::endl;
        return false;
    }

    //if(!isfinite(normal.length())) {
    //    std::cout << "error, triangle:" << this << std::endl
    //            << ", normal.length() is not finite" << std::endl;
    //    return false;
    //}

    return true;
}


bool MxPartialPolygon::isValid() const
{
    assert(polygon);
    int id = (&polygon->partialTriangles[0] == this) ? 0 : 1;

    if(polygon->cells[id]->isRoot()) {
        if(mass != 0.) {
            std::cout << "error, partial triangle id:"
                    << polygon->id << "." << id
                    << ", mass for root cell partial triangle must be zero"
                    << std::endl;
            return false;
        }
    } else {
        if(!isfinite(mass)) {
            std::cout << "error, partial triangle id:"
                    << polygon->id << "." << id
                    << ", mass is not finite"
                    << std::endl;
            return false;
        }
    }

    return true;
}


float MxPolygon::volume(CCellPtr cell) const
{
    assert(cells[0] == cell || cells[1] == cell);
    float direction = cells[0] == cell ? 1.0f : -1.0f;

    return direction * _volume;
}


