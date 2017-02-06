/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File: Shape2DNode.cpp
*
******************************************************************/

#include <x3d/Shape2DNode.h>

using namespace CyberX3D;

static const char appearanceExposedFieldName[] = "appearance";
static const char geometryExposedFieldName[] = "geometry";

Shape2DNode::Shape2DNode() 
{
	setHeaderFlag(false);
	setType(SHAPE2D_NODE);

	// appearance field
	appField = new SFNode();
	addExposedField(appearanceExposedFieldName, appField);

	// geometry field
	geomField = new SFNode();
	addExposedField(geometryExposedFieldName, geomField);
}

Shape2DNode::~Shape2DNode() 
{
}

////////////////////////////////////////////////
//	Appearance
////////////////////////////////////////////////

SFNode *Shape2DNode::getAppearanceField() const
{
	if (isInstanceNode() == false)
		return appField;
	return (SFNode *)getExposedField(appearanceExposedFieldName);
}

////////////////////////////////////////////////
//	Geometry
////////////////////////////////////////////////

SFNode *Shape2DNode::getGeometryField() const
{
	if (isInstanceNode() == false)
		return geomField;
	return (SFNode *)getExposedField(geometryExposedFieldName);
}

////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

Shape2DNode *Shape2DNode::next() const
{
	return (Shape2DNode *)Node::next(getType());
}

Shape2DNode *Shape2DNode::nextTraversal() const
{
	return (Shape2DNode *)Node::nextTraversalByType(getType());
}

////////////////////////////////////////////////
//	functions
////////////////////////////////////////////////
	
bool Shape2DNode::isChildNodeType(Node *node) const
{
	return true;
}

void Shape2DNode::initialize() 
{
}

void Shape2DNode::uninitialize() 
{
}

void Shape2DNode::update() 
{
}

////////////////////////////////////////////////
//	Infomation
////////////////////////////////////////////////

void Shape2DNode::outputContext(std::ostream &printStream, const char *indentString) const
{
}

