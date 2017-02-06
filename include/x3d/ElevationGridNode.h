/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	ElevationGridNode.h
*
******************************************************************/

#ifndef _CX3D_ELEVATIONGRIDNODE_H_
#define _CX3D_ELEVATIONGRIDNODE_H_

#include <x3d/ComposedGeometryNode.h>
#include <x3d/ColorNode.h>
#include <x3d/NormalNode.h>
#include <x3d/TextureCoordinateNode.h>

namespace CyberX3D {

class ElevationGridNode : public ComposedGeometryNode 
{
	SFFloat *xSpacingField;
	SFFloat *zSpacingField;
	SFInt32 *xDimensionField;
	SFInt32 *zDimensionField;
	SFFloat *creaseAngleField;
	MFFloat *heightField;

public:

	ElevationGridNode();
	virtual ~ElevationGridNode();

	////////////////////////////////////////////////
	//	xSpacing
	////////////////////////////////////////////////

	SFFloat *getXSpacingField() const;

	void setXSpacing(float value);
	float getXSpacing() const;

	////////////////////////////////////////////////
	//	zSpacing
	////////////////////////////////////////////////

	SFFloat *getZSpacingField() const;

	void setZSpacing(float value);
	float getZSpacing() const;

	////////////////////////////////////////////////
	//	xDimension
	////////////////////////////////////////////////

	SFInt32 *getXDimensionField() const;

	void setXDimension(int value);
	int getXDimension() const;

	////////////////////////////////////////////////
	//	zDimension
	////////////////////////////////////////////////

	SFInt32 *getZDimensionField() const;

	void setZDimension(int value);
	int getZDimension() const;

	////////////////////////////////////////////////
	//	CreaseAngle
	////////////////////////////////////////////////

	SFFloat *getCreaseAngleField() const;
	
	void setCreaseAngle(float value);
	float getCreaseAngle() const;

	////////////////////////////////////////////////
	// height
	////////////////////////////////////////////////

	MFFloat *getHeightField() const;

	void addHeight(float value);
	int getNHeights() const;
	float getHeight(int index) const;

	////////////////////////////////////////////////
	//	List
	////////////////////////////////////////////////

	ElevationGridNode *next() const;
	ElevationGridNode *nextTraversal() const;

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

	void recomputeBoundingBox();

	////////////////////////////////////////////////
	//	recomputeDisplayList
	////////////////////////////////////////////////

	void recomputeDisplayList();

	////////////////////////////////////////////////
	//	Polygons
	////////////////////////////////////////////////

	int getNPolygons() const;

	////////////////////////////////////////////////
	//	Infomation
	////////////////////////////////////////////////

	void outputContext(std::ostream &printStream, const char *indentString) const;
};

}

#endif

