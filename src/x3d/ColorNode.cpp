/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File: ColorNode.cpp
*
*	Revisions:
*
*	11/25/02
*		- Changed the super class from Node to GeometricPropertyNode.
*
******************************************************************/

#include <x3d/VRML97Fields.h>
#include <x3d/ColorNode.h>

using namespace CyberX3D;

ColorNode::ColorNode() 
{
	setHeaderFlag(false);
	setType(COLOR_NODE);

	// color exposed field
	colorField = new MFColor();
	colorField->setName(colorFieldString);
	addExposedField(colorField);
}

ColorNode::~ColorNode() 
{
}

////////////////////////////////////////////////
//	color
////////////////////////////////////////////////

MFColor *ColorNode::getColorField() const
{
	if (isInstanceNode() == false)
		return colorField;
	return (MFColor *)getExposedField(colorFieldString);
}
	
void ColorNode::addColor(float color[])
{
	getColorField()->addValue(color);
}

int ColorNode::getNColors() const 
{
	return getColorField()->getSize();
}

void ColorNode::getColor(int index, float color[]) const 
{
	getColorField()->get1Value(index, color);
}

////////////////////////////////////////////////
//	functions
////////////////////////////////////////////////
	
bool ColorNode::isChildNodeType(Node *node) const
{
	return false;
}

void ColorNode::initialize() 
{
}

void ColorNode::uninitialize() 
{
}

void ColorNode::update() 
{
}

////////////////////////////////////////////////
//	Output
////////////////////////////////////////////////

void ColorNode::outputContext(std::ostream &printStream, const char *indentString) const 
{
	if (0 < getNColors()) { 
		MFColor *color = getColorField();
		printStream <<  indentString << "\t" << "color ["  << std::endl;
		color->MField::outputContext(printStream, indentString, "\t\t");
		printStream << indentString << "\t" << "]" << std::endl;
	}
}

////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

ColorNode *ColorNode::next() const 
{
	return (ColorNode *)Node::next(getType());
}

ColorNode *ColorNode::nextTraversal() const 
{
	return (ColorNode *)Node::nextTraversalByType(getType());
}

