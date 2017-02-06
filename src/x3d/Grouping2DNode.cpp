/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	Grouping2DNode.cpp
*
******************************************************************/

#include <x3d/Grouping2DNode.h>
#include <x3d/SFVec2f.h>

using namespace CyberX3D;

static const char addChildrenEventIn[] = "addChildren";
static const char removeChildrenEventIn[] = "removeChildren";
static const char bboxCenterFieldName[] = "bboxCenter";
static const char bboxSizeFieldName[] = "bboxSize";

Grouping2DNode::Grouping2DNode() 
{
	setHeaderFlag(false);

/*
	// addChildren eventout field
	MFNode addNodes = new MFNode();
	addEventIn(addChildrenEventIn, addNodes);

	// removeChildren eventout field
	MFNode removeChildren = new MFNode();
	addEventIn(removeChildrenEventIn, removeChildren);
*/

	// bboxCenter field
	bboxCenterField = new SFVec2f(0.0f, 0.0f);
	bboxCenterField->setName(bboxCenterFieldName);
	addField(bboxCenterField);

	// bboxSize field
	bboxSizeField = new SFVec2f(-1.0f, -1.0f);
	bboxSizeField->setName(bboxSizeFieldName);
	addField(bboxSizeField);
}

Grouping2DNode::~Grouping2DNode() 
{
}

////////////////////////////////////////////////
//	BoundingBoxSize
////////////////////////////////////////////////

SFVec2f *Grouping2DNode::getBoundingBoxSizeField() const
{
	if (isInstanceNode() == false)
		return bboxSizeField;
	return (SFVec2f *)getField(bboxSizeFieldName);
}

void Grouping2DNode::setBoundingBoxSize(float value[]) 
{
	getBoundingBoxSizeField()->setValue(value);
}

void Grouping2DNode::setBoundingBoxSize(float x, float y) 
{
	getBoundingBoxSizeField()->setValue(x, y);
}

void Grouping2DNode::getBoundingBoxSize(float value[])  const
{
	getBoundingBoxSizeField()->getValue(value);
}

////////////////////////////////////////////////
//	BoundingBoxCenter
////////////////////////////////////////////////

SFVec2f *Grouping2DNode::getBoundingBoxCenterField() const
{
	if (isInstanceNode() == false)
		return bboxCenterField;
	return (SFVec2f *)getField(bboxCenterFieldName);
}

void Grouping2DNode::setBoundingBoxCenter(float value[]) 
{
	getBoundingBoxCenterField()->setValue(value);
}

void Grouping2DNode::setBoundingBoxCenter(float x, float y) 
{
	getBoundingBoxCenterField()->setValue(x, y);
}

void Grouping2DNode::getBoundingBoxCenter(float value[]) const
{
	getBoundingBoxCenterField()->getValue(value);
}

////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

Grouping2DNode *Grouping2DNode::next() const
{
/*
	for (Node *node = Node::next(); node != NULL; node = node->next()) {
		if (node->isGroupNode() || node->isTransformNode() || node->isBillboardNode() || node->isCollisionNode() || node->isLODNode() || node->isSwitchNode() || node->isInlineNode())
			return (Grouping2DNode *)node;
	}
*/
	return NULL;
}

Grouping2DNode *Grouping2DNode::nextTraversal() const
{
/*
	for (Node *node = Node::nextTraversal(); node != NULL; node = node->nextTraversal()) {
		if (node->isGroupNode() || node->isTransformNode() || node->isBillboardNode() || node->isCollisionNode() || node->isLODNode() || node->isSwitchNode() || node->isInlineNode())
			return (Grouping2DNode *)node;
	}
*/
	return NULL;
}
