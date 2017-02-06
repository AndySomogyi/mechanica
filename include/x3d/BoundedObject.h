/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	BoundedObject.h
*
*	Revisions:
*		11/18/02
*			- The first release.
*
******************************************************************/

#ifndef _CX3D_BOUNDEDOBJECT_H_
#define _CX3D_BOUNDEDOBJECT_H_

namespace CyberX3D {

class BoundingBox;

class BoundedObject
{

public:

	BoundedObject()
	{
	}

	virtual ~BoundedObject()
	{
	}

	////////////////////////////////////////////////
	//	BoundingBoxSize
	////////////////////////////////////////////////

	virtual SFVec3f *getBoundingBoxSizeField() const = 0;

	virtual void setBoundingBoxSize(float value[]) = 0;
	virtual void setBoundingBoxSize(float x, float y, float z) = 0;
	virtual void getBoundingBoxSize(float value[]) const = 0;

	////////////////////////////////////////////////
	//	BoundingBoxCenter
	////////////////////////////////////////////////

	virtual SFVec3f *getBoundingBoxCenterField() const = 0;

	virtual void setBoundingBoxCenter(float value[]) = 0;
	virtual void setBoundingBoxCenter(float x, float y, float z) = 0;
	virtual void getBoundingBoxCenter(float value[]) const = 0;

	////////////////////////////////////////////////
	//	BoundingBox
	////////////////////////////////////////////////

	virtual void setBoundingBox(BoundingBox *bbox) = 0;
	virtual void updateBoundingBox() = 0;
};

}

#endif

