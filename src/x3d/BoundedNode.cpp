/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	BoundedNode.cpp
*
*	Revisions:
*
*		11/15/02
*			- The first release.
*
******************************************************************/

#include <float.h>
#include <x3d/Geometry3DNode.h>
#include <x3d/BoundedNode.h>

using namespace CyberX3D;

static const char bboxCenterFieldName[] = "bboxCenter";
static const char bboxSizeFieldName[] = "bboxSize";

BoundedNode::BoundedNode() 
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

BoundedNode::~BoundedNode() 
{
}

////////////////////////////////////////////////
//	BoundingBoxSize
////////////////////////////////////////////////

SFVec3f *BoundedNode::getBoundingBoxSizeField() const
{
	if (isInstanceNode() == false)
		return bboxSizeField;
	return (SFVec3f *)getField(bboxSizeFieldName);
}

void BoundedNode::setBoundingBoxSize(float value[]) 
{
	getBoundingBoxSizeField()->setValue(value);
}

void BoundedNode::setBoundingBoxSize(float x, float y, float z) 
{
	getBoundingBoxSizeField()->setValue(x, y, z);
}

void BoundedNode::getBoundingBoxSize(float value[])  const
{
	getBoundingBoxSizeField()->getValue(value);
}

////////////////////////////////////////////////
//	BoundingBoxCenter
////////////////////////////////////////////////

SFVec3f *BoundedNode::getBoundingBoxCenterField() const
{
	if (isInstanceNode() == false)
		return bboxCenterField;
	return (SFVec3f *)getField(bboxCenterFieldName);
}

void BoundedNode::setBoundingBoxCenter(float value[]) 
{
	getBoundingBoxCenterField()->setValue(value);
}

void BoundedNode::setBoundingBoxCenter(float x, float y, float z) 
{
	getBoundingBoxCenterField()->setValue(x, y, z);
}

void BoundedNode::getBoundingBoxCenter(float value[])  const
{
	getBoundingBoxCenterField()->getValue(value);
}

////////////////////////////////////////////////
//	BoundingBox
////////////////////////////////////////////////

void BoundedNode::setBoundingBox(BoundingBox *bbox) 
{
	float center[3];
	float size[3];
	bbox->getCenter(center);
	bbox->getSize(size);
	setBoundingBoxCenter(center);
	setBoundingBoxSize(size);
}

////////////////////////////////////////////////
//	BoundedNode::recomputeBoundingBox
////////////////////////////////////////////////

void CyberX3D::UpdateBoundingBox(
Node		*node,
BoundingBox	*bbox)
{
	if (node->isGeometry3DNode()) {
		Geometry3DNode *gnode = (Geometry3DNode *)node;
		gnode->recomputeBoundingBox();

		float	bboxCenter[3];
		float	bboxSize[3];

		gnode->getBoundingBoxCenter(bboxCenter);
		gnode->getBoundingBoxSize(bboxSize);

		SFMatrix	mx;
		gnode->getTransformMatrix(&mx);

		for (int n=0; n<8; n++) {
			float	point[3];
			point[0] = (n < 4)			? bboxCenter[0] - bboxSize[0] : bboxCenter[0] + bboxSize[0];
			point[1] = (n % 2)			? bboxCenter[1] - bboxSize[1] : bboxCenter[1] + bboxSize[1];
			point[2] = ((n % 4) < 2)	? bboxCenter[2] - bboxSize[2] : bboxCenter[2] + bboxSize[2];
			mx.multi(point);
			bbox->addPoint(point);
		}
	}

	for (Node *cnode=node->getChildNodes(); cnode; cnode=cnode->next()) 
		UpdateBoundingBox(cnode, bbox);
}

void BoundedNode::updateBoundingBox()
{
	BoundingBox bbox;

	for (Node *node=getChildNodes(); node; node=node->next()) 
		UpdateBoundingBox(node, &bbox);

	setBoundingBox(&bbox);
}

