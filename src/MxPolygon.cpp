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
#include "MxEdge.h"

static MxType partialPolygonType{"MxPartialPolygon", MxObject_Type};
MxType *MxPartialPolygon_Type = &partialPolygonType;

static MxPolygonType polygonType{"MxPolygonType", MxObject_Type};
MxPolygonType *MxPolygon_Type = &polygonType;

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
    << "cells:{" << (tri->cells[0] ? to_string(tri->cells[0]->id) : "null") << "," << (tri->cells[1] ? to_string(tri->cells[1]->id) : "null") << "}," << std::endl
       << "vertices:{";
       for(CVertexPtr v : tri->vertices) {
           os << v->id << ", ";
       }
    os << "}" << std::endl << "edges: {" << std::endl;
    for(CEdgePtr e : tri->edges) {
        os << "\t" << e <<  ", " << std::endl;
    }
    os << "}" << std::endl;

    return os;
}


MxPolygon::MxPolygon(uint _id, MxType* type) :
            id{_id}, MxObject{type}, cells{{nullptr, nullptr}},
            partialPolygons{{{MxPartialPolygon_Type, this}, {MxPartialPolygon_Type, this}}} {


    edges.resize(vertices.size());
    _vertexNormals.resize(vertices.size());
                _vertexAreas.resize(vertices.size());

    positionsChanged();
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

    if(vertices.size() < 3) {
        return S_OK;
    }

    // triangle area is 1/2 * length of surface normal (non-normalized)

    for (int i = 0; i < vertices.size(); ++i) {
        int prevIndex = loopIndex(i-1, vertices.size());
        int nextIndex = loopIndex(i+1, vertices.size());

        assert(prevIndex != nextIndex);

        CVertexPtr vp = vertices[prevIndex];
        CVertexPtr v = vertices[i];
        CVertexPtr vn = vertices[nextIndex];

        checkVec(vp->position);
        checkVec(v->position);
        checkVec(vn->position);

        Vector3 np = triangleNormal((v->position + vp->position) / 2., v->position, centroid);
        Vector3 nn = triangleNormal(v->position, (vn->position + v->position) / 2., centroid);
        Vector3 vertNormal = (np + nn);
        float vertLen = vertNormal.length();

        checkVec(np);
        checkVec(nn);

        // normalized surface normals and vertex areas
        _vertexNormals[i] = vertNormal / vertLen;
        _vertexAreas[i] = 0.5 * vertLen;

        checkVec(_vertexNormals[i]);

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
    /*
    float vertArea = 0;
    for (float v :_vertexAreas) {
        vertArea += v;
    }

    std::cout << "vert area: " << vertArea << ", poly area: " << area << ", diff: " << area - vertArea << std::endl;

    Vector3 vertNormal;
    for(const Vector3& v : _vertexNormals) {
        vertNormal += v;
    }

    vertNormal /= _vertexNormals.size();

    std::cout << "normal: " << normal << ", vertex normal: " << vertNormal << ", dot: " << Math::dot(normal, vertNormal) << std::endl;
     */
#endif

    assert(area >= 0);

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
            assert(partialPolygons[i].mass == 0.);
        } else {
	  std::isfinite(partialPolygons[i].mass ) && partialPolygons[i].mass  > 0;
        }
    }

    if(!isConnected()) {
        std::cout << "error, triangle:" << this << std::endl
                << " is not connected" << std::endl;
        return false;
    }

    if(!std::isfinite(area)) {
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

    if(!std::isfinite(getMass())) {
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
    int id = (&polygon->partialPolygons[0] == this) ? 0 : 1;

    if(polygon->cells[id]->isRoot()) {
        if(mass != 0.) {
            std::cout << "error, partial triangle id:"
                    << polygon->id << "." << id
                    << ", mass for root cell partial triangle must be zero"
                    << std::endl;
            return false;
        }
    } else {
      if(!std::isfinite(mass)) {
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

int MxPolygon::vertexIndex(CVertexPtr vert) const {
    for(int i = 0; i < vertices.size(); ++i) {
        if(vertices[i] == vert) {
            return i;
        }
    }
    return -1;
}

bool MxPolygon::checkEdges() const
{
    for(int i = 0; i < vertices.size(); ++i) {
        if(!edges[i]->matches(vertices[i], vertices[(i+1) % vertices.size()])) {
            std::cout << "edge " << i << " in polygon " << this->id << " does not match verts" << std::endl;
            return false;
        }
        if(!connectedEdgePolygonPointers(edges[i], this)) {
            std::cout << "edge " << edges[i] << " is in polygon, but not connected" << std::endl;
            return false;
        }
    }
    return true;
}
