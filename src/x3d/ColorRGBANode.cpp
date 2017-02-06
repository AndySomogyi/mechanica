/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File: ColorRGBANode.cpp
*
*	Revisions:
*
*	11/25/02
*		- The first revision.
*
******************************************************************/

#include <x3d/X3DFields.h>
#include <x3d/ColorRGBANode.h>

using namespace CyberX3D;

ColorRGBANode::ColorRGBANode() 
{
	setHeaderFlag(false);
	setType(COLORRGBA_NODE);

	// color exposed field
	colorField = new MFColorRGBA();
	colorField->setName(colorFieldString);
	addExposedField(colorField);
}

ColorRGBANode::~ColorRGBANode() 
{
}

////////////////////////////////////////////////
//	color
////////////////////////////////////////////////

MFColorRGBA *ColorRGBANode::getColorField() const
{
	if (isInstanceNode() == false)
		return colorField;
	return (MFColorRGBA *)getExposedField(colorFieldString);
}
	
void ColorRGBANode::addColor(float color[]) 
{
	getColorField()->addValue(color);
}

int ColorRGBANode::getNColors() const 
{
	return getColorField()->getSize();
}

void ColorRGBANode::getColor(int index, float color[]) const 
{
	getColorField()->get1Value(index, color);
}

////////////////////////////////////////////////
//	functions
////////////////////////////////////////////////
	
bool ColorRGBANode::isChildNodeType(Node *node) const
{
	return false;
}

void ColorRGBANode::initialize() 
{
}

void ColorRGBANode::uninitialize() 
{
}

void ColorRGBANode::update() 
{
}

////////////////////////////////////////////////
//	Output
////////////////////////////////////////////////

void ColorRGBANode::outputContext(std::ostream &printStream, const char *indentString) const 
{
	if (0 < getNColors()) { 
		MFColorRGBA *color = getColorField();
		printStream <<  indentString << "\t" << "color ["  << std::endl;
		color->MField::outputContext(printStream, indentString, "\t\t");
		printStream << indentString << "\t" << "]" << std::endl;
	}
}

////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

ColorRGBANode *ColorRGBANode::next() const 
{
	return (ColorRGBANode *)Node::next(getType());
}

ColorRGBANode *ColorRGBANode::nextTraversal() const 
{
	return (ColorRGBANode *)Node::nextTraversalByType(getType());
}

