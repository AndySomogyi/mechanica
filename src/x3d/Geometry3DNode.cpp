/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	Geometry3DNode.cpp
*
******************************************************************/

#include <x3d/Geometry3DNode.h>

using namespace CyberX3D;

Geometry3DNode::Geometry3DNode() 
{
	// bboxCenter field
	bboxCenterField = new SFVec3f(0.0f, 0.0f, 0.0f);
	bboxCenterField->setName(bboxCenterPrivateFieldName);
	addPrivateField(bboxCenterField);

	// bboxSize field
	bboxSizeField = new SFVec3f(-1.0f, -1.0f, -1.0f);
	bboxSizeField->setName(bboxSizePrivateFieldName);
	addPrivateField(bboxSizeField);

	setBoundingBoxCenter(0.0f, 0.0f, 0.0f);
	setBoundingBoxSize(-1.0f, -1.0f, -1.0f);
}

Geometry3DNode::~Geometry3DNode() 
{
}

////////////////////////////////////////////////
//	BoundingBoxSize
////////////////////////////////////////////////

SFVec3f *Geometry3DNode::getBoundingBoxSizeField() const
{
	if (isInstanceNode() == false)
		return bboxSizeField;
	return (SFVec3f *)getPrivateField(bboxSizePrivateFieldName);
}

void Geometry3DNode::setBoundingBoxSize(float value[]) 
{
	getBoundingBoxSizeField()->setValue(value);
}

void Geometry3DNode::setBoundingBoxSize(float x, float y, float z) 
{
	getBoundingBoxSizeField()->setValue(x, y, z);
}

void Geometry3DNode::getBoundingBoxSize(float value[]) const
{
	getBoundingBoxSizeField()->getValue(value);
}

////////////////////////////////////////////////
//	BoundingBoxCenter
////////////////////////////////////////////////

SFVec3f *Geometry3DNode::getBoundingBoxCenterField() const
{
	if (isInstanceNode() == false)
		return bboxCenterField;
	return (SFVec3f *)getPrivateField(bboxCenterPrivateFieldName);
}

void Geometry3DNode::setBoundingBoxCenter(float value[]) 
{
	getBoundingBoxCenterField()->setValue(value);
}

void Geometry3DNode::setBoundingBoxCenter(float x, float y, float z) 
{
	getBoundingBoxCenterField()->setValue(x, y, z);
}

void Geometry3DNode::getBoundingBoxCenter(float value[]) const
{
	getBoundingBoxCenterField()->getValue(value);
}

////////////////////////////////////////////////
//	BoundingBox
////////////////////////////////////////////////

void Geometry3DNode::setBoundingBox(BoundingBox *bbox) 
{
	float center[3];
	float size[3];
	bbox->getCenter(center);
	bbox->getSize(size);
	setBoundingBoxCenter(center);
	setBoundingBoxSize(size);
}
