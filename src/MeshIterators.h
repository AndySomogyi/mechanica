/*
 * MeshIterators.h
 *
 *  Created on: Aug 13, 2017
 *      Author: andy
 */

#ifndef SRC_MESHITERATORS_H_
#define SRC_MESHITERATORS_H_

#include "MxMeshCore.h"
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

    size_t size() const;

    explicit EdgeTriangles(TrianglePtr startTri, int index);

    explicit EdgeTriangles(const Edge& edge);

    explicit EdgeTriangles() {};

#ifndef NDEBUG
    bool isValid();
#endif

private:
    std::vector<TrianglePtr> triangles;
    friend class EdgeTriangleIterator;
};

/**
 * grab all of the triangles in a fan.
 */
std::vector<TrianglePtr> triangleFan(CVertexPtr vert, CCellPtr cell);


#endif /* SRC_MESHITERATORS_H_ */
