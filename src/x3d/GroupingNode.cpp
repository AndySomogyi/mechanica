/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	GroupingNode.cpp
*
*	Revisions:
*		11/15/02
*			- Exported the following fields to BoundedNode.
*				bboxCenter, bboxSize
*
******************************************************************/

#include <float.h>
#include <x3d/X3DFields.h>
#include <x3d/GroupingNode.h>
#include <x3d/Geometry3DNode.h>

using namespace CyberX3D;

static const char addChildrenEventIn[] = "addChildren";
static const char removeChildrenEventIn[] = "removeChildren";

GroupingNode::GroupingNode() 
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
}

GroupingNode::~GroupingNode() 
{
}

////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

GroupingNode *GroupingNode::next() const
{
	for (Node *node = Node::next(); node != NULL; node = node->next()) {
		if (node->isGroupNode() || node->isTransformNode() || node->isBillboardNode() || node->isCollisionNode() || node->isLODNode() || node->isSwitchNode() || node->isInlineNode())
			return (GroupingNode *)node;
	}
	return NULL;
}

GroupingNode *GroupingNode::nextTraversal() const
{
	for (Node *node = Node::nextTraversal(); node != NULL; node = node->nextTraversal()) {
		if (node->isGroupNode() || node->isTransformNode() || node->isBillboardNode() || node->isCollisionNode() || node->isLODNode() || node->isSwitchNode() || node->isInlineNode())
			return (GroupingNode *)node;
	}
	return NULL;
}

