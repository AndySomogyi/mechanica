/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	Bounded2DObject.h
*
*	Revisions:
*		12/02/02
*			- The first release.
*
******************************************************************/

#ifndef _CX3D_BOUNDED2DOBJECT_H_
#define _CX3D_BOUNDED2DOBJECT_H_

namespace CyberX3D {

class BoundingBox2D;

class Bounded2DObject
{

public:

	Bounded2DObject()
	{
	}

	virtual ~Bounded2DObject()
	{
	}

	////////////////////////////////////////////////
	//	BoundingBoxSize
	////////////////////////////////////////////////

	virtual SFVec2f *getBoundingBoxSizeField() const = 0;

	virtual void setBoundingBoxSize(float value[]) = 0;
	virtual void setBoundingBoxSize(float x, float y) = 0;
	virtual void getBoundingBoxSize(float value[]) const = 0;

	////////////////////////////////////////////////
	//	BoundingBoxCenter
	////////////////////////////////////////////////

	virtual SFVec2f *getBoundingBoxCenterField() const = 0;

	virtual void setBoundingBoxCenter(float value[]) = 0;
	virtual void setBoundingBoxCenter(float x, float y) = 0;
	virtual void getBoundingBoxCenter(float value[]) const = 0;

	////////////////////////////////////////////////
	//	BoundingBox
	////////////////////////////////////////////////

	virtual void setBoundingBox(BoundingBox2D *bbox) = 0;
	virtual void updateBoundingBox() = 0;
};

}

#endif

