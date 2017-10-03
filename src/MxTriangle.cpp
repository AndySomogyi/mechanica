/*
 * MxTriangle.cpp
 *
 *  Created on: Oct 3, 2017
 *      Author: andy
 */

#include "MxTriangle.h"
#include "MxCell.h"


int MxTriangle::matchVertexIndices(const std::array<VertexPtr, 3> &indices) {
    typedef std::array<VertexPtr, 3> vertind;

    if (vertices == indices ||
        vertices == vertind{{indices[1], indices[2], indices[0]}} ||
        vertices == vertind{{indices[2], indices[0], indices[1]}}) {
        return 1;
    }

    if (vertices == vertind{{indices[2], indices[1], indices[0]}} ||
        vertices == vertind{{indices[1], indices[0], indices[2]}} ||
        vertices == vertind{{indices[0], indices[2], indices[1]}}) {
        return -1;
    }
    return 0;
}



float MxTriangle::aspectRatio() const {
	const Vector3& v1 = vertices[0]->position;
	const Vector3& v2 = vertices[1]->position;
	const Vector3& v3 = vertices[2]->position;

    float a = (v1 - v2).length();
    float b = (v2 - v3).length();
    float c = (v3 - v1).length();
    float s = (a + b + c) / 2.0;
    return (a * b * c) / (8.0 * (s - a) * (s - b) * (s - c));
}

HRESULT MxTriangle::attachToCell(CellPtr cell)  {

	if(cells[0] == nullptr) {
		cells[0] = cell;

        // make sure the tri is not already in the cell.
		assert(std::find(cell->boundary.begin(), cell->boundary.end(),
				&partialTriangles[0]) == cell->boundary.end());

		cell->boundary.push_back(&partialTriangles[0]);
		return S_OK;
	}
	if(cells[1] == nullptr) {
		cells[1] = cell;

		assert(std::find(cell->boundary.begin(), cell->boundary.end(),
						&partialTriangles[1]) == cell->boundary.end());

		cell->boundary.push_back(&partialTriangles[1]);
		return S_OK;
	}
	return E_FAIL;
}

MxTriangle::MxTriangle(MxTriangleType* type,
		const std::array<VertexPtr, 3>& verts,
		const std::array<CellPtr, 2>& cells,
		const std::array<MxPartialTriangleType*, 2>& partTriTypes,
		FacetPtr facet) :
			MxObject{type}, vertices{verts}, cells{cells},
			partialTriangles{{{partTriTypes[0], this}, {partTriTypes[1], this}}},
			facet{facet} {
}

