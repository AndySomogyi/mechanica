/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	IndexedLinSet.h
*
******************************************************************/

#ifndef _CX3D_INDEXEDLINESETNODE_H_
#define _CX3D_INDEXEDLINESETNODE_H_

#include <x3d/GeometryNode.h>
#include <x3d/ColorNode.h>
#include <x3d/CoordinateNode.h>

namespace CyberX3D {

class IndexedLineSetNode : public GeometryNode {

	SFBool *colorPerVertexField;
	MFInt32 *coordIndexField;
	MFInt32 *colorIndexField;
	SFFloat *lineWidthField;

	MFInt32 *set_coordIndexField;
	MFInt32 *set_colorIndexField;

	SFNode *colorField;
	SFNode *coordField;
	
public:

	IndexedLineSetNode();
	virtual ~IndexedLineSetNode();

	////////////////////////////////////////////////
	//	Color
	////////////////////////////////////////////////

	SFNode *getColorField() const;
	
	////////////////////////////////////////////////
	//	Coord
	////////////////////////////////////////////////

	SFNode *getCoordField() const;

	////////////////////////////////////////////////
	// set_coordIndex
	////////////////////////////////////////////////

	MFInt32 *getSetCoordIndexField() const;

	////////////////////////////////////////////////
	// set_colorIndex
	////////////////////////////////////////////////

	MFInt32 *getSetColorIndexField() const;
	
	////////////////////////////////////////////////
	//	ColorPerVertex
	////////////////////////////////////////////////

	SFBool *getColorPerVertexField() const;
	
	void setColorPerVertex(bool value);
	void setColorPerVertex(int value);
	bool getColorPerVertex() const;

	////////////////////////////////////////////////
	// CoordIndex
	////////////////////////////////////////////////

	MFInt32 *getCoordIndexField() const;

	void addCoordIndex(int value);
	int getNCoordIndexes() const;
	int getCoordIndex(int index) const;
	void clearCoordIndex();
	
	////////////////////////////////////////////////
	// ColorIndex
	////////////////////////////////////////////////

	MFInt32 *getColorIndexField() const;

	void addColorIndex(int value);
	int getNColorIndexes() const;
	int getColorIndex(int index) const;
	void clearColorIndex();

	////////////////////////////////////////////////
	//	LineWidth
	////////////////////////////////////////////////

	SFFloat *getLineWidthField() const;
	void setLineWidth(float value);
	float getLineWidth() const;

	////////////////////////////////////////////////
	//	List
	////////////////////////////////////////////////

	IndexedLineSetNode *next() const;
	IndexedLineSetNode *nextTraversal() const;

	////////////////////////////////////////////////
	//	functions
	////////////////////////////////////////////////
	
	bool isChildNodeType(Node *node) const;
	void initialize();
	void uninitialize();
	void update();

	////////////////////////////////////////////////
	//	BoundingBox
	////////////////////////////////////////////////

	//void recomputeBoundingBox();

	////////////////////////////////////////////////
	//	Polygons
	////////////////////////////////////////////////

	int getNPolygons() const {
		return 0;
	}

	////////////////////////////////////////////////
	//	recomputeDisplayList
	////////////////////////////////////////////////

#ifdef CX3D_SUPPORT_OPENGL
	void recomputeDisplayList();
#endif

	////////////////////////////////////////////////
	//	Infomation
	////////////////////////////////////////////////

	void outputContext(std::ostream &printStream, const char *indentString) const;
};

}

#endif
