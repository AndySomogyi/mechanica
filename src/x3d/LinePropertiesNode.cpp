/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File: LinePropertiesNode.h
*
*	Revisions:
*
*	12/02/02
*		- The first revision.
*
******************************************************************/

#include <x3d/LinePropertiesNode.h>

using namespace CyberX3D;

static const char lineStyleFieldString[] = "lineStyle";
static const char widthFieldString[] = "width";

LinePropertiesNode::LinePropertiesNode() 
{
	setHeaderFlag(false);
	setType(LINEPROPERTIES_NODE);

	// lineStyle exposed field
	lineStyleField = new SFInt32(0);
	lineStyleField->setName(lineStyleFieldString);
	addExposedField(lineStyleField);

	// width exposed field
	widthField = new SFFloat(0.0f);
	widthField->setName(widthFieldString);
	addExposedField(widthField);
}

LinePropertiesNode::~LinePropertiesNode() 
{
}

////////////////////////////////////////////////
//	LineStyle
////////////////////////////////////////////////

SFInt32 *LinePropertiesNode::getLineStyleField() const
{
	if (isInstanceNode() == false)
		return lineStyleField;
	return (SFInt32 *)getExposedField(lineStyleFieldString);
}
	
void LinePropertiesNode::setLineStyle(int value) 
{
	getLineStyleField()->setValue(value);
}

int LinePropertiesNode::getLineStyle() const
{
	return getLineStyleField()->getValue();
}

////////////////////////////////////////////////
//	Width
////////////////////////////////////////////////

SFFloat *LinePropertiesNode::getWidthField() const
{
	if (isInstanceNode() == false)
		return widthField;
	return (SFFloat *)getExposedField(widthFieldString);
}
	
void LinePropertiesNode::setWidth(float value) 
{
	getWidthField()->setValue(value);
}

float LinePropertiesNode::getWidth() const
{
	return getWidthField()->getValue();
}

////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

LinePropertiesNode *LinePropertiesNode::next() const
{
	return (LinePropertiesNode *)Node::next(getType());
}

LinePropertiesNode *LinePropertiesNode::nextTraversal() const
{
	return (LinePropertiesNode *)Node::nextTraversalByType(getType());
}

////////////////////////////////////////////////
//	functions
////////////////////////////////////////////////
	
bool LinePropertiesNode::isChildNodeType(Node *node) const
{
	return false;
}

void LinePropertiesNode::initialize() 
{
}

void LinePropertiesNode::uninitialize() 
{
}

void LinePropertiesNode::update() 
{
}

////////////////////////////////////////////////
//	Infomation
////////////////////////////////////////////////

void LinePropertiesNode::outputContext(std::ostream &printStream, const char *indentString) const
{
}
