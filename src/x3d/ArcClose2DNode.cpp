/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	ArcClose2DNode.cpp
*
******************************************************************/

#include <x3d/ArcClose2DNode.h>
#include <x3d/Graphic3D.h>

using namespace CyberX3D;

static const char closureTypeFieldString[] = "closureType";

ArcClose2DNode::ArcClose2DNode() 
{
	setHeaderFlag(false);
	setType(ARCCLOSE2D_NODE);

	///////////////////////////
	// Exposed Field 
	///////////////////////////

	// closureType exposed field
	closureTypeField = new SFString("PIE");
	addField(closureTypeFieldString, closureTypeField);
}

ArcClose2DNode::~ArcClose2DNode() 
{
}

////////////////////////////////////////////////
//	ClosureType
////////////////////////////////////////////////

SFString *ArcClose2DNode::getClosureTypeField() const
{
	if (isInstanceNode() == false)
		return closureTypeField;
	return (SFString *)getField(closureTypeFieldString);
}
	
void ArcClose2DNode::setClosureType(const char *value) 
{
	getClosureTypeField()->setValue(value);
}

const char *ArcClose2DNode::getClosureType() const
{
	return getClosureTypeField()->getValue();
} 

////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

ArcClose2DNode *ArcClose2DNode::next() const
{
	return (ArcClose2DNode *)Node::next(getType());
}

ArcClose2DNode *ArcClose2DNode::nextTraversal() const
{
	return (ArcClose2DNode *)Node::nextTraversalByType(getType());
}

////////////////////////////////////////////////
//	functions
////////////////////////////////////////////////
	
bool ArcClose2DNode::isChildNodeType(Node *node) const
{
	return false;
}

void ArcClose2DNode::initialize() 
{
#ifdef CX3D_SUPPORT_OPENGL
	recomputeDisplayList();
#endif
}

void ArcClose2DNode::uninitialize() 
{
}

void ArcClose2DNode::update() 
{
}

////////////////////////////////////////////////
//	outputContext
////////////////////////////////////////////////

void ArcClose2DNode::outputContext(std::ostream &printStream, const char *indentString) const
{
}

////////////////////////////////////////////////
//	ArcClose2DNode::recomputeDisplayList
////////////////////////////////////////////////

#ifdef CX3D_SUPPORT_OPENGL

void ArcClose2DNode::recomputeDisplayList() 
{
};

#endif
