/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	Bounded2DNode.cpp
*
*	Revisions:
*
*		11/15/02
*			- The first release.
*
******************************************************************/

#include <float.h>
#include <x3d/Geometry3DNode.h>
#include <x3d/Bounded2DNode.h>

using namespace CyberX3D;

static const char bboxCenterFieldName[] = "bboxCenter";
static const char bboxSizeFieldName[] = "bboxSize";

Bounded2DNode::Bounded2DNode() 
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

Bounded2DNode::~Bounded2DNode() 
{
}

////////////////////////////////////////////////
//	BoundingBox2DSize
////////////////////////////////////////////////

SFVec2f *Bounded2DNode::getBoundingBoxSizeField() const
{
	if (isInstanceNode() == false)
		return bboxSizeField;
	return (SFVec2f *)getField(bboxSizeFieldName);
}

void Bounded2DNode::setBoundingBoxSize(float value[]) 
{
	getBoundingBoxSizeField()->setValue(value);
}

void Bounded2DNode::setBoundingBoxSize(float x, float y) 
{
	getBoundingBoxSizeField()->setValue(x, y);
}

void Bounded2DNode::getBoundingBoxSize(float value[])  const
{
	getBoundingBoxSizeField()->getValue(value);
}

////////////////////////////////////////////////
//	BoundingBox2DCenter
////////////////////////////////////////////////

SFVec2f *Bounded2DNode::getBoundingBoxCenterField() const
{
	if (isInstanceNode() == false)
		return bboxCenterField;
	return (SFVec2f *)getField(bboxCenterFieldName);
}

void Bounded2DNode::setBoundingBoxCenter(float value[]) 
{
	getBoundingBoxCenterField()->setValue(value);
}

void Bounded2DNode::setBoundingBoxCenter(float x, float y) 
{
	getBoundingBoxCenterField()->setValue(x, y);
}

void Bounded2DNode::getBoundingBoxCenter(float value[])  const
{
	getBoundingBoxCenterField()->getValue(value);
}

////////////////////////////////////////////////
//	BoundingBox2D
////////////////////////////////////////////////

void Bounded2DNode::setBoundingBox(BoundingBox2D *bbox) 
{
	float center[3];
	float size[3];
	bbox->getCenter(center);
	bbox->getSize(size);
	setBoundingBoxCenter(center);
	setBoundingBoxSize(size);
}

////////////////////////////////////////////////
//	Bounded2DNode::recomputeBoundingBox2D
////////////////////////////////////////////////

void CyberX3D::UpdateBoundingBox2D(
Node		*node,
BoundingBox2D	*bbox)
{
/*
	if (node->isGeometry3DNode()) {
		Bounded2DNode *b2dNode = (Bounded2DNode *)node;
		b2dNode->recomputeBoundingBox2D();

		float	bboxCenter[3];
		float	bboxSize[3];

		b2dNode->getBoundingBoxCenter(bboxCenter);
		b2dNode->getBoundingBoxSize(bboxSize);

		SFMatrix	mx;
		b2dNode->getTransformMatrix(&mx);

		for (int n=0; n<4; n++) {
			float	point[3];
			point[0] = (n < 4)			? bboxCenter[0] - bboxSize[0] : bboxCenter[0] + bboxSize[0];
			point[1] = (n % 2)			? bboxCenter[1] - bboxSize[1] : bboxCenter[1] + bboxSize[1];
			mx.multi(point);
			bbox->addPoint(point);
		}
	}
*/

	for (Node *cnode=node->getChildNodes(); cnode; cnode=cnode->next()) 
		UpdateBoundingBox2D(cnode, bbox);
}

void Bounded2DNode::updateBoundingBox()
{
	BoundingBox2D bbox;

	for (Node *node=getChildNodes(); node; node=node->next()) 
		UpdateBoundingBox2D(node, &bbox);

	setBoundingBox(&bbox);
}

