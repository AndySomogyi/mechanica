/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	TriangleSet2DNode.cpp
*
******************************************************************/

#include <x3d/TriangleSet2DNode.h>
#include <x3d/Graphic3D.h>

using namespace CyberX3D;

static const char verticesFieldString[] = "vertices";

TriangleSet2DNode::TriangleSet2DNode() 
{
	setHeaderFlag(false);
	setType(TRIANGLESET2D_NODE);

	///////////////////////////
	// Field 
	///////////////////////////

	// vertices field
	verticesField = new MFVec2f();
	addField(verticesFieldString, verticesField);
}

TriangleSet2DNode::~TriangleSet2DNode() 
{
}

////////////////////////////////////////////////
//	Vertices
////////////////////////////////////////////////

MFVec2f *TriangleSet2DNode::getVerticesField() const
{
	if (isInstanceNode() == false)
		return verticesField;
	return 	(MFVec2f *)getField(verticesFieldString);
}

int TriangleSet2DNode::getNVertices() const
{
	return getVerticesField()->getSize();
}

void TriangleSet2DNode::addVertex(float point[]) 
{
	getVerticesField()->addValue(point);
}

void TriangleSet2DNode::addVertex(float x, float y) 
{
	getVerticesField()->addValue(x, y);
}

void TriangleSet2DNode::getVertex(int index, float point[]) const
{
	getVerticesField()->get1Value(index, point);
}

void TriangleSet2DNode::setVertex(int index, float point[]) 
{
	getVerticesField()->set1Value(index, point);
}

void TriangleSet2DNode::setVertex(int index, float x, float y) 
{
	getVerticesField()->set1Value(index, x, y);
}

void TriangleSet2DNode::removeVertex(int index) 
{
	getVerticesField()->remove(index);
}

void TriangleSet2DNode::removeAllVertices() 
{
	getVerticesField()->clear();
}

////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

TriangleSet2DNode *TriangleSet2DNode::next() const
{
	return (TriangleSet2DNode *)Node::next(getType());
}

TriangleSet2DNode *TriangleSet2DNode::nextTraversal() const
{
	return (TriangleSet2DNode *)Node::nextTraversalByType(getType());
}

////////////////////////////////////////////////
//	functions
////////////////////////////////////////////////
	
bool TriangleSet2DNode::isChildNodeType(Node *node) const
{
	return false;
}

void TriangleSet2DNode::initialize() 
{
	recomputeBoundingBox();
#ifdef CX3D_SUPPORT_OPENGL
		recomputeDisplayList();
#endif
}

void TriangleSet2DNode::uninitialize() 
{
}

void TriangleSet2DNode::update() 
{
}

////////////////////////////////////////////////
//	Infomation
////////////////////////////////////////////////

void TriangleSet2DNode::outputContext(std::ostream &printStream, const char *indentString) const
{
}

////////////////////////////////////////////////
//	TriangleSet2DNode::recomputeDisplayList
////////////////////////////////////////////////

#ifdef CX3D_SUPPORT_OPENGL

void TriangleSet2DNode::recomputeDisplayList() 
{
};

#endif
