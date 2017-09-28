/*
 * MeshIterators.h
 *
 *  Created on: Aug 13, 2017
 *      Author: andy
 */

#ifndef SRC_MESHITERATORS_H_
#define SRC_MESHITERATORS_H_

#include <cstddef>
#include <array>





/**
 * Enumerates all of the triangles that share an edge.
 */
class TriangleEdgeStarIterator {
public:
    // Iterator traits, previously from std::iterator.
    using value_type = MxTriangle;
    using difference_type = std::ptrdiff_t;
    using pointer = MxTriangle*;
    using reference = MxTriangle&;
    using iterator_category = std::bidirectional_iterator_tag;

    // Default constructible.
    TriangleEdgeStarIterator() = default;
    explicit TriangleEdgeStarIterator(const class EdgeTriangles &edgeStar);

    // Dereferencable.
    reference operator*() const;

    // Pre- and post-incrementable.
    TriangleEdgeStarIterator& operator++();
    TriangleEdgeStarIterator operator++(int);

    // Pre- and post-decrementable.
    TriangleEdgeStarIterator& operator--();
    TriangleEdgeStarIterator operator--(int);

    // Equality / inequality.
    bool operator==(const TriangleEdgeStarIterator& rhs);
    bool operator!=(const TriangleEdgeStarIterator& rhs);

private:

};


class EdgeTriangles {
public:
  using value_type = MxTriangle;



public:

  using const_iterator = TriangleEdgeStarIterator;

  const_iterator begin() const;

  const_iterator end() const;

  explicit EdgeTriangles(const MxMesh* mesh, const MxTriangle &startTri, const std::array<VertexPtr, 2> &edge);


};


/**
 * Enumerates all of the triangles that share an edge.
 */
class EdgeTrianglesIterator {
public:
    // Iterator traits, previously from std::iterator.
    using value_type = MxCell;
    using difference_type = std::ptrdiff_t;
    using pointer = MxCell*;
    using reference = MxCell&;
    using iterator_category = std::bidirectional_iterator_tag;

    // Default constructible.
    EdgeTrianglesIterator() = default;
    explicit EdgeTrianglesIterator(const class EdgeTriangles &edgeStar);

    // Dereferencable.
    reference operator*() const;

    // Pre- and post-incrementable.
    TriangleEdgeStarIterator& operator++();
    TriangleEdgeStarIterator operator++(int);

    // Pre- and post-decrementable.
    TriangleEdgeStarIterator& operator--();
    TriangleEdgeStarIterator operator--(int);

    // Equality / inequality.
    bool operator==(const TriangleEdgeStarIterator& rhs);
    bool operator!=(const TriangleEdgeStarIterator& rhs);

private:

};





/**
 * Enumerates all of the triangles that share an edge.
 */
class EdgeFacetIterator {
public:
    // Iterator traits, previously from std::iterator.
    using value_type = FacetPtr;
    using difference_type = std::ptrdiff_t;
    using pointer = FacetPtr*;
    using reference = FacetPtr&;
    using iterator_category = std::bidirectional_iterator_tag;

    // Default constructible.
    EdgeFacetIterator() = default;
    explicit EdgeFacetIterator(const class EdgeFacets &edgeStar);

    // Dereferencable.
    reference operator*() const;

    // Pre- and post-incrementable.
    TriangleEdgeStarIterator& operator++();
    TriangleEdgeStarIterator operator++(int);

    // Pre- and post-decrementable.
    TriangleEdgeStarIterator& operator--();
    TriangleEdgeStarIterator operator--(int);

    // Equality / inequality.
    bool operator==(const EdgeFacetIterator& rhs);
    bool operator!=(const EdgeFacetIterator& rhs);

private:

};

class EdgeFacets {
public:
  using value_type = FacetPtr;



public:

  using const_iterator = EdgeFacetIterator;

  const_iterator begin() const;

  const_iterator end() const;

  explicit EdgeFacets(const TrianglePtr startTri, const std::array<VertexPtr, 2> &edge);
};



#if 0

/**
 * Enumerates all of the triangles that share an edge.
 */
class VertexTrianglesIterator {
public:
    // Iterator traits, previously from std::iterator.
    using value_type = MxTriangle;
    using difference_type = std::ptrdiff_t;
    using pointer = MxTriangle*;
    using reference = MxTriangle&;
    using iterator_category = std::forward_iterator_tag;

    // Default constructVertexTrianglesIteratorlesIterator() = delete;
    VertexTrianglesIterator(MxMesh *mesh, VertexTriangleIndx vertexTriangleId);

    // Dereferencable.
    reference operator*() const;

    // Pre- and post-incrementable.
    VertexTrianglesIterator& operator++();
    VertexTrianglesIterator operator++(int);

    // Pre- and post-decrementable.
    VertexTrianglesIterator& operator--();
    VertexTrianglesIterator operator--(int);

    // Equality / inequality.
    bool operator==(const VertexTrianglesIterator& rhs);
    bool operator!=(const VertexTrianglesIterator& rhs);

private:

    MxMesh *mesh;
    MxVertexTriangle vertexTriangle;
};


class VertexTriangles {
public:

    using iterator = VertexTrianglesIterator;

    iterator begin() const {
        return iterator{mesh, startId};
    }

    iterator end() const {
        return iterator{nullptr, 0};
    }

    explicit VertexTriangles(MxMesh* mesh, const MxVertex& vertex) :
        mesh{mesh}, startId{vertex.trianglesId} {}

private:
    MxMesh *mesh;
    VertexTriangleIndx startId;
};

inline VertexTriangles triangles(MxMesh *mesh, const MxVertex &vertex) {
    return VertexTriangles{mesh, vertex};
}

#endif



#endif /* SRC_MESHITERATORS_H_ */
