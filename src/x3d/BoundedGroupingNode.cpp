/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File: BoundedGroupingNode.cpp
*
*	Revisions:
*		11/19/02
*			- The first release.
*
******************************************************************/

#include <float.h>
#include <x3d/Geometry3DNode.h>
#include <x3d/BoundedGroupingNode.h>

using namespace CyberX3D;

static const char bboxCenterFieldName[] = "bboxCenter";
static const char bboxSizeFieldName[] = "bboxSize";

BoundedGroupingNode::BoundedGroupingNode() 
{
	setHeaderFlag(false);

	// bboxCenter field
	bboxCenterField = new SFVec3f(0.0f, 0.0f, 0.0f);
	bboxCenterField->setName(bboxCenterFieldName);
	addField(bboxCenterField);

	// bboxSize field
	bboxSizeField = new SFVec3f(-1.0f, -1.0f, -1.0f);
	bboxSizeField->setName(bboxSizeFieldName);
	addField(bboxSizeField);
}

BoundedGroupingNode::~BoundedGroupingNode() 
{
}

////////////////////////////////////////////////
//	BoundingBoxSize
////////////////////////////////////////////////

SFVec3f *BoundedGroupingNode::getBoundingBoxSizeField() const
{
	if (isInstanceNode() == false)
		return bboxSizeField;
	return (SFVec3f *)getField(bboxSizeFieldName);
}

void BoundedGroupingNode::setBoundingBoxSize(float value[]) 
{
	getBoundingBoxSizeField()->setValue(value);
}

void BoundedGroupingNode::setBoundingBoxSize(float x, float y, float z) 
{
	getBoundingBoxSizeField()->setValue(x, y, z);
}

void BoundedGroupingNode::getBoundingBoxSize(float value[])  const
{
	getBoundingBoxSizeField()->getValue(value);
}

////////////////////////////////////////////////
//	BoundingBoxCenter
////////////////////////////////////////////////

SFVec3f *BoundedGroupingNode::getBoundingBoxCenterField() const
{
	if (isInstanceNode() == false)
		return bboxCenterField;
	return (SFVec3f *)getField(bboxCenterFieldName);
}

void BoundedGroupingNode::setBoundingBoxCenter(float value[]) 
{
	getBoundingBoxCenterField()->setValue(value);
}

void BoundedGroupingNode::setBoundingBoxCenter(float x, float y, float z) 
{
	getBoundingBoxCenterField()->setValue(x, y, z);
}

void BoundedGroupingNode::getBoundingBoxCenter(float value[])  const
{
	getBoundingBoxCenterField()->getValue(value);
}

////////////////////////////////////////////////
//	BoundingBox
////////////////////////////////////////////////

void BoundedGroupingNode::setBoundingBox(BoundingBox *bbox) 
{
	float center[3];
	float size[3];
	bbox->getCenter(center);
	bbox->getSize(size);
	setBoundingBoxCenter(center);
	setBoundingBoxSize(size);
}

////////////////////////////////////////////////
//	BoundedGroupingNode::updateBoundingBox
////////////////////////////////////////////////

void BoundedGroupingNode::updateBoundingBox()
{
	BoundingBox bbox;

	for (Node *node=getChildNodes(); node; node=node->next()) 
		UpdateBoundingBox(node, &bbox);

	setBoundingBox(&bbox);
}

