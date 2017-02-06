/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	FogNode.cpp
*
******************************************************************/

#include <x3d/FogNode.h>

using namespace CyberX3D;

FogNode::FogNode() 
{
	setHeaderFlag(false);
	setType(FOG_NODE);

	///////////////////////////
	// Exposed Field 
	///////////////////////////
		
	// color exposed field
	colorField = new SFColor(1.0f, 1.0f, 1.0f);
	addExposedField(colorFieldString, colorField);

	// fogType exposed field
	fogTypeField = new SFString("LINEAR");
	addExposedField(fogTypeFieldString, fogTypeField);

	// visibilityRange exposed field
	visibilityRangeField = new SFFloat(0.0f);
	addExposedField(visibilityRangeFieldString, visibilityRangeField);
}

FogNode::~FogNode() 
{
}

////////////////////////////////////////////////
//	Color
////////////////////////////////////////////////

SFColor *FogNode::getColorField() const
{
	if (isInstanceNode() == false)
		return colorField;
	return (SFColor *)getExposedField(colorFieldString);
}

void FogNode::setColor(float value[])
{
	getColorField()->setValue(value);
}

void FogNode::setColor(float r, float g, float b) 
{
	getColorField()->setValue(r, g, b);
}

void FogNode::getColor(float value[]) const
{
	getColorField()->getValue(value);
}

////////////////////////////////////////////////
//	FogType
////////////////////////////////////////////////

SFString *FogNode::getFogTypeField() const
{
	if (isInstanceNode() == false)
		return fogTypeField;
	return (SFString *)getExposedField(fogTypeFieldString);
}

void FogNode::setFogType(const char *value) 
{
	getFogTypeField()->setValue(value);
}

const char *FogNode::getFogType() const
{
	return getFogTypeField()->getValue();
}

////////////////////////////////////////////////
//	VisibilityRange
////////////////////////////////////////////////

SFFloat *FogNode::getVisibilityRangeField() const
{
	if (isInstanceNode() == false)
		return visibilityRangeField;
	return (SFFloat *)getExposedField(visibilityRangeFieldString);
}

void FogNode::setVisibilityRange(float value) 
{
	getVisibilityRangeField()->setValue(value);
}

float FogNode::getVisibilityRange() const
{
	return getVisibilityRangeField()->getValue();
}

////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

FogNode *FogNode::next() const
{
	return (FogNode *)Node::next(getType());
}

FogNode *FogNode::nextTraversal() const
{
	return (FogNode *)Node::nextTraversalByType(getType());
}

////////////////////////////////////////////////
//	functions
////////////////////////////////////////////////
	
bool FogNode::isChildNodeType(Node *node) const
{
	return false;
}

void FogNode::initialize() 
{
}

void FogNode::uninitialize() 
{
}

void FogNode::update() 
{
}

////////////////////////////////////////////////
//	Infomation
////////////////////////////////////////////////

void FogNode::outputContext(std::ostream &printStream, const char *indentString) const
{
	SFColor *color = getColorField();
	SFString *fogType = getFogTypeField();

	printStream << indentString << "\t" << "color " << color << std::endl;
	printStream << indentString << "\t" << "fogType " << fogType << std::endl;
	printStream << indentString << "\t" << "visibilityRange " << getVisibilityRange() << std::endl;
}
