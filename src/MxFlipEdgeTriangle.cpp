/*
 * HITransition.cpp
 *
 *  Created on: Jan 16, 2019
 *      Author: andy
 */

#include "MeshOperations.h"

#define __fastcall
#include "DirectXMath.h"



bool Mx_IsTriangleToEdgeConfiguration(CEdgePtr edge) {
    return false;
}

static bool isEdgeToTriangleConfiguration(CEdgePtr edge, CCellPtr *edgeCells, CCellPtr *c0, CCellPtr *c1) {

    // make sure we have 2 vertices and 3 polys
    if(edge->vertexCount() != 2 || edge->polygonCount() != 3) {
        return false;
    }

    // make sure each poly has at least 4 sides
    for(uint i = 0; i < 3; ++i) {
        if(edge->polygons[i]->edges.size() < 4) {
            return false;
        }
    }

    // cells for 0 and 1 vertices.
    std::set<CCellPtr> cells, cells0, cells1;

    // cells around the edge
    for(uint i = 0; i < 3; ++i) {
        cells.insert(edge->polygons[i]->cells[0]);
        cells.insert(edge->polygons[i]->cells[1]);
    }

    assert(cells.size() == 3);
    
    //for(CCellPtr c : cells) {
    //    std::cout << "cell around edge: " << c->id << std::endl;
    //}

    // grab the cells at the top and bottom of the edge
    for(uint i = 0; i < 3; ++i) {
        CPolygonPtr poly = edge->polygons[i];
        int edgeIndex = poly->edgeIndex(edge);
        int prevIndex = mod(edgeIndex-1, poly->edges.size());
        int nextIndex = mod(edgeIndex+1, poly->edges.size());

        CEdgePtr e = poly->edges[prevIndex];
        for(uint j = 0; j < e->polygonCount(); ++j) {
            CPolygonPtr p = e->polygons[j];
            
            //std::cout << "cells in polygon " << j << " for prev edge {" << p->cells[0]->id << ", " << p->cells[1]->id << "}" << std::endl;
            
            if(p->vertexIndex(edge->vertices[0]) >= 0) {
                if(cells.find(p->cells[0]) == cells.end()) {
                    cells0.insert(p->cells[0]);
                }
                if(cells.find(p->cells[1]) == cells.end()) {
                    cells0.insert(p->cells[1]);
                }
            }
            if(p->vertexIndex(edge->vertices[1]) >= 0) {
                if(cells.find(p->cells[0]) == cells.end()) {
                    cells1.insert(p->cells[0]);
                }

                if(cells.find(p->cells[1]) == cells.end()) {
                    cells1.insert(p->cells[1]);
                }
            }
        }
        
        e = poly->edges[nextIndex];
        for(uint j = 0; j < e->polygonCount(); ++j) {
            CPolygonPtr p = e->polygons[j];
            if(p->vertexIndex(edge->vertices[0]) >= 0) {
                if(cells.find(p->cells[0]) == cells.end()) {
                    cells0.insert(p->cells[0]);
                }
                if(cells.find(p->cells[1]) == cells.end()) {
                    cells0.insert(p->cells[1]);
                }
            }
            if(p->vertexIndex(edge->vertices[1]) >= 0) {
                if(cells.find(p->cells[0]) == cells.end()) {
                    cells1.insert(p->cells[0]);
                }
                
                if(cells.find(p->cells[1]) == cells.end()) {
                    cells1.insert(p->cells[1]);
                }
            }
        }
    }
    
    if(cells1.size() == 0 || cells0.size() == 0) {
        return false;
    }

    assert(cells1.size() == 1);
    assert(cells0.size() == 1);

    CCellPtr cell0 = *cells0.begin();
    CCellPtr cell1 = *cells1.begin();
    
    if(edgeCells) {
        int i = 0;
        for(CCellPtr c : cells) {
            edgeCells[i++] = c;
        }
    }
    
    if(c0) {
        *c0 = cell0;
    }
    
    if(c1) {
        *c1 = cell1;
    }

    return (cell0 != cell1);
}

bool Mx_IsEdgeToTriangleConfiguration(CEdgePtr edge) {
    return isEdgeToTriangleConfiguration(edge, nullptr, nullptr, nullptr);
}

HRESULT Mx_FlipTriangleToEdge(MeshPtr mesh, PolygonPtr poly, EdgePtr* edge)
{
    return E_NOTIMPL;
}

HRESULT Mx_FlipEdgeToTriangle(MeshPtr mesh, EdgePtr edge, PolygonPtr* poly)
{
    return E_NOTIMPL;
}
