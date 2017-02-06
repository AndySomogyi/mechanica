/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	Geometry3DNode.h
*
******************************************************************/

#ifndef _CX3D_GEOMETRY3DNODE_H_
#define _CX3D_GEOMETRY3DNODE_H_

#include <x3d/X3DFields.h>
#include <x3d/GeometryNode.h>
#include <x3d/BoundingBox.h>

namespace CyberX3D {

const char bboxCenterPrivateFieldName[] = "bboxCenter";
const char bboxSizePrivateFieldName[] = "bboxSize";

class Geometry3DNode : public GeometryNode 
{

	SFVec3f *bboxCenterField;
	SFVec3f *bboxSizeField;

	int slices;

public:

	Geometry3DNode();
	virtual ~Geometry3DNode();

	////////////////////////////////////////////////
	//	BoundingBoxSize
	////////////////////////////////////////////////

	SFVec3f *getBoundingBoxSizeField() const;

	void setBoundingBoxSize(float value[]);
	void setBoundingBoxSize(float x, float y, float z);
	void getBoundingBoxSize(float value[]) const;

	////////////////////////////////////////////////
	//	BoundingBoxCenter
	////////////////////////////////////////////////

	SFVec3f *getBoundingBoxCenterField() const;

	void setBoundingBoxCenter(float value[]);
	void setBoundingBoxCenter(float x, float y, float z);
	void getBoundingBoxCenter(float value[]) const;

	////////////////////////////////////////////////
	//	BoundingBox
	////////////////////////////////////////////////

	void setBoundingBox(BoundingBox *bbox);

	////////////////////////////////////////////////
	//	Slices
	////////////////////////////////////////////////

	void setSlices(int value)
	{
		slices = value;
	}

	int getSlices() const 
	{
		return slices;
	}

	////////////////////////////////////////////////
	//	Polygons
	////////////////////////////////////////////////

	virtual int getNPolygons() const = 0;
};

}

#endif
