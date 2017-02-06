/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File: BoundedGrouping2DNode.cpp
*
*	Revisions:
*		11/19/02
*			- The first release.
*
******************************************************************/

#include <float.h>
#include <x3d/BoundedGrouping2DNode.h>
#include <x3d/SFVec2f.h>
#include <x3d/Bounded2DNode.h>

using namespace CyberX3D;

static const char bboxCenterFieldName[] = "bboxCenter";
static const char bboxSizeFieldName[] = "bboxSize";

BoundedGrouping2DNode::BoundedGrouping2DNode() 
{
	setHeaderFlag(false);

	// bboxCenter field
	bboxCenterField = new SFVec2f(0.0f, 0.0f);
	bboxCenterField->setName(bboxCenterFieldName);
	addField(bboxCenterField);

	// bboxSize field
	bboxSizeField = new SFVec2f(-1.0f, -1.0f);
	bboxSizeField->setName(bboxSizeFieldName);
	addField(bboxSizeField);
}

BoundedGrouping2DNode::~BoundedGrouping2DNode() 
{
}

////////////////////////////////////////////////
//	BoundingBoxSize
////////////////////////////////////////////////

SFVec2f *BoundedGrouping2DNode::getBoundingBoxSizeField() const
{
	if (isInstanceNode() == false)
		return bboxSizeField;
	return (SFVec2f *)getField(bboxSizeFieldName);
}

void BoundedGrouping2DNode::setBoundingBoxSize(float value[]) 
{
	getBoundingBoxSizeField()->setValue(value);
}

void BoundedGrouping2DNode::setBoundingBoxSize(float x, float y) 
{
	getBoundingBoxSizeField()->setValue(x, y);
}

void BoundedGrouping2DNode::getBoundingBoxSize(float value[])  const
{
	getBoundingBoxSizeField()->getValue(value);
}

////////////////////////////////////////////////
//	BoundingBoxCenter
////////////////////////////////////////////////

SFVec2f *BoundedGrouping2DNode::getBoundingBoxCenterField() const
{
	if (isInstanceNode() == false)
		return bboxCenterField;
	return (SFVec2f *)getField(bboxCenterFieldName);
}

void BoundedGrouping2DNode::setBoundingBoxCenter(float value[]) 
{
	getBoundingBoxCenterField()->setValue(value);
}

void BoundedGrouping2DNode::setBoundingBoxCenter(float x, float y) 
{
	getBoundingBoxCenterField()->setValue(x, y);
}

void BoundedGrouping2DNode::getBoundingBoxCenter(float value[])  const
{
	getBoundingBoxCenterField()->getValue(value);
}

////////////////////////////////////////////////
//	BoundingBox2D
////////////////////////////////////////////////

void BoundedGrouping2DNode::setBoundingBox(BoundingBox2D *bbox) 
{
	float center[3];
	float size[3];
	bbox->getCenter(center);
	bbox->getSize(size);
	setBoundingBoxCenter(center);
	setBoundingBoxSize(size);
}

////////////////////////////////////////////////
//	BoundedGrouping2DNode::updateBoundingBox
////////////////////////////////////////////////

void BoundedGrouping2DNode::updateBoundingBox()
{
	BoundingBox2D bbox;

	for (Node *node=getChildNodes(); node; node=node->next()) 
		UpdateBoundingBox2D(node, &bbox);

	setBoundingBox(&bbox);
}

