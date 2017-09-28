/*
 * MeshIterators.cpp
 *
 *  Created on: Aug 13, 2017
 *      Author: andy
 */

#include <MxCell.h>

TriangleEdgeStarIterator::TriangleEdgeStarIterator(
		const class EdgeTriangles& edgeStar) {
}

TriangleEdgeStarIterator::reference TriangleEdgeStarIterator::operator *() const {
}

TriangleEdgeStarIterator& TriangleEdgeStarIterator::operator ++() {
}

TriangleEdgeStarIterator TriangleEdgeStarIterator::operator ++(int int1) {
}

TriangleEdgeStarIterator& TriangleEdgeStarIterator::operator --() {
}

TriangleEdgeStarIterator TriangleEdgeStarIterator::operator --(int int1) {
}

bool TriangleEdgeStarIterator::operator ==(
		const TriangleEdgeStarIterator& rhs) {
}

bool TriangleEdgeStarIterator::operator !=(
		const TriangleEdgeStarIterator& rhs) {
}

EdgeTriangles::const_iterator EdgeTriangles::begin() const {
}

EdgeTriangles::const_iterator EdgeTriangles::end() const {
}

EdgeTriangles::EdgeTriangles(const MxMesh* mesh, const MxTriangle& startTri,
		const std::array<VertexPtr, 2>& edge) {
}

EdgeTrianglesIterator::EdgeTrianglesIterator(
		const class EdgeTriangles& edgeStar) {
}

EdgeTrianglesIterator::reference EdgeTrianglesIterator::operator *() const {
}

TriangleEdgeStarIterator& EdgeTrianglesIterator::operator ++() {
}

TriangleEdgeStarIterator EdgeTrianglesIterator::operator ++(int int1) {
}

TriangleEdgeStarIterator& EdgeTrianglesIterator::operator --() {
}

TriangleEdgeStarIterator EdgeTrianglesIterator::operator --(int int1) {
}

bool EdgeTrianglesIterator::operator ==(const TriangleEdgeStarIterator& rhs) {
}

bool EdgeTrianglesIterator::operator !=(const TriangleEdgeStarIterator& rhs) {
}

EdgeFacetIterator::EdgeFacetIterator(const class EdgeFacets& edgeStar) {
}

EdgeFacetIterator::reference EdgeFacetIterator::operator *() const {
}

TriangleEdgeStarIterator& EdgeFacetIterator::operator ++() {
}

TriangleEdgeStarIterator EdgeFacetIterator::operator ++(int int1) {
}

TriangleEdgeStarIterator& EdgeFacetIterator::operator --() {
}

TriangleEdgeStarIterator EdgeFacetIterator::operator --(int int1) {
}

bool EdgeFacetIterator::operator ==(const EdgeFacetIterator& rhs) {
}

bool EdgeFacetIterator::operator !=(const EdgeFacetIterator& rhs) {
}

EdgeFacets::const_iterator EdgeFacets::begin() const {
}

EdgeFacets::const_iterator EdgeFacets::end() const {
}

EdgeFacets::EdgeFacets(const TrianglePtr startTri,
		const std::array<VertexPtr, 2>& edge) {
}
