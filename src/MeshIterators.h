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
class EdgeTriangleIterator {
public:
    // Iterator traits, previously from std::iterator.
    using value_type = TrianglePtr;
    using difference_type = std::ptrdiff_t;
    using pointer = TrianglePtr*;
    using reference = TrianglePtr&;
    using iterator_category = std::bidirectional_iterator_tag;

    // Default constructible.
    EdgeTriangleIterator() = delete;

    // Dereferencable.
    value_type operator*() const;

    // pre-incrementable.
    EdgeTriangleIterator& operator++();

    // post increment
    EdgeTriangleIterator operator++(int);

    // Pre- and post-decrementable.
    EdgeTriangleIterator& operator--();
    EdgeTriangleIterator operator--(int);

    // Equality / inequality.
    bool operator==(const EdgeTriangleIterator& rhs);
    bool operator!=(const EdgeTriangleIterator& rhs);

private:
    explicit EdgeTriangleIterator(const std::vector<TrianglePtr> &triangles, int index) :
        triangles{triangles}, index{index} {};

    const std::vector<TrianglePtr> &triangles;
    int index;

    friend class EdgeTriangles;
};

/**
 * Enumerate triangles that share an edge.
 *
 * This is a temporary measure to enumerate triangles based on the triangle lists
 * of a pair of vertices. This is very slow, will switch over to triangle linked
 * lists. This class will hide the linked list implementation.
 */
class EdgeTriangles {
public:
    using value_type = TrianglePtr;

    using iterator = EdgeTriangleIterator;

    iterator begin() const;

    iterator end() const;

    explicit EdgeTriangles(TrianglePtr startTri, int index);

private:
    std::vector<TrianglePtr> triangles;
    friend class EdgeTriangleIterator;
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
    EdgeTriangleIterator& operator++();
    EdgeTriangleIterator operator++(int);

    // Pre- and post-decrementable.
    EdgeTriangleIterator& operator--();
    EdgeTriangleIterator operator--(int);

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
