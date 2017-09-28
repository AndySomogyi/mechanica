/*
 * MxEdge.cpp
 *
 *  Created on: Sep 27, 2017
 *      Author: andy
 */

#include "MxCell.h"
#include "MxEdge.h"

MxEdge::MxEdge(VertexPtr a, VertexPtr b) {
}

MxEdge::MxEdge(const TrianglePtr a, const TrianglePtr b) {
}

EdgeFacets MxEdge::facets() const {
}

std::vector<TrianglePtr> MxEdge::radialTriangles() const {
}

const MxEdge::FacetVector& MxEdge::upperFacets() const {
}

const MxEdge::FacetVector& MxEdge::lowerFacets() const {
}

const MxEdge::FacetVector& MxEdge::radialFacets() const {
}

bool MxEdge::operator ==(const MxEdge& other) {
}

bool MxEdge::incidentTo(const MxTriangle& tri) {
}
