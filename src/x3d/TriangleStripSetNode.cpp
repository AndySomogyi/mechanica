/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	TriangleStripSetNode.cpp
*
*	Revisions;
*
*	11/27/02
*		- The first revision.
*
******************************************************************/

#include <x3d/TriangleStripSetNode.h>
#include <x3d/Graphic3D.h>

using namespace CyberX3D;

static const char stripCountFieldString[] = "stripCount";

TriangleStripSetNode::TriangleStripSetNode() 
{
	setHeaderFlag(false);
	setType(TRIANGLESTRIPSET_NODE);

	///////////////////////////
	// Field 
	///////////////////////////

	// stripCount  field
	stripCountField = new MFInt32();
	stripCountField->setName(stripCountFieldString);
	addField(stripCountField);
}

TriangleStripSetNode::~TriangleStripSetNode() 
{
}
	
////////////////////////////////////////////////
// StripCount
////////////////////////////////////////////////

MFInt32 *TriangleStripSetNode::getStripCountField() const
{
	if (isInstanceNode() == false)
		return stripCountField;
	return (MFInt32 *)getField(stripCountFieldString);
}

void TriangleStripSetNode::addStripCount(int value) 
{
	getStripCountField()->addValue(value);
}

int TriangleStripSetNode::getNStripCountes() const
{
	return getStripCountField()->getSize();
}

int TriangleStripSetNode::getStripCount(int index) const
{
	return getStripCountField()->get1Value(index);
}

////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

TriangleStripSetNode *TriangleStripSetNode::next() const
{
	return (TriangleStripSetNode *)Node::next(getType());
}

TriangleStripSetNode *TriangleStripSetNode::nextTraversal() const
{
	return (TriangleStripSetNode *)Node::nextTraversalByType(getType());
}

////////////////////////////////////////////////
//	functions
////////////////////////////////////////////////
	
bool TriangleStripSetNode::isChildNodeType(Node *node) const
{
	if (node->isColorNode() || node->isCoordinateNode() || node->isNormalNode() || node->isTextureCoordinateNode())
		return true;
	return false;
}

void TriangleStripSetNode::uninitialize() 
{
}

void TriangleStripSetNode::update() 
{
}

////////////////////////////////////////////////
//	Infomation
////////////////////////////////////////////////

void TriangleStripSetNode::outputContext(std::ostream &printStream, const char *indentString) const
{
}

////////////////////////////////////////////////
//	TriangleStripSetNode::initialize
////////////////////////////////////////////////

void TriangleStripSetNode::initialize() 
{
}

////////////////////////////////////////////////////////////
//	getNPolygons
////////////////////////////////////////////////////////////

int TriangleStripSetNode::getNPolygons() const
{
	int nStrips = getNStripCountes();
	return nStrips - 2;
}

////////////////////////////////////////////////
//	recomputeDisplayList
////////////////////////////////////////////////

#ifdef CX3D_SUPPORT_OPENGL

void TriangleStripSetNode::recomputeDisplayList() 
{
}

#endif


