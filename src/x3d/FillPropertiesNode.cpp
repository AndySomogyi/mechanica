/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File: FillPropertiesNode.h
*
*	Revisions:
*
*	12/02/02
*		- The first revision.
*
******************************************************************/

#include <x3d/FillPropertiesNode.h>

using namespace CyberX3D;

static const char fillStyleFieldString[] = "fillStyle";
static const char hatchStyleFieldString[] = "hatchStyle";
static const char hatchColorFieldString[] = "hatchColor";

FillPropertiesNode::FillPropertiesNode() 
{
	setHeaderFlag(false);
	setType(FILLPROPERTIES_NODE);

	// fillStyle exposed field
	fillStyleField = new SFString("NONE");
	fillStyleField->setName(fillStyleFieldString);
	addExposedField(fillStyleField);

	// hatchStyle exposed field
	hatchStyleField = new SFInt32(1);
	hatchStyleField->setName(hatchStyleFieldString);
	addExposedField(hatchStyleField);

	// hatchColor exposed field
	hatchColorField = new SFColor(1.0f, 1.0f, 1.0f);
	hatchColorField->setName(hatchColorFieldString);
	addExposedField(hatchColorField);
}

FillPropertiesNode::~FillPropertiesNode() 
{
}

////////////////////////////////////////////////
//	FillStyle
////////////////////////////////////////////////

SFString *FillPropertiesNode::getFillStyleField() const
{
	if (isInstanceNode() == false)
		return fillStyleField;
	return (SFString *)getExposedField(fillStyleFieldString);
}
	
void FillPropertiesNode::setFillStyle(const char *value) 
{
	getFillStyleField()->setValue(value);
}

const char *FillPropertiesNode::getFillStyle() const
{
	return getFillStyleField()->getValue();
}

////////////////////////////////////////////////
//	HatchStyle
////////////////////////////////////////////////

SFInt32 *FillPropertiesNode::getHatchStyleField() const
{
	if (isInstanceNode() == false)
		return hatchStyleField;
	return (SFInt32 *)getExposedField(hatchStyleFieldString);
}
	
void FillPropertiesNode::setHatchStyle(int value) 
{
	getHatchStyleField()->setValue(value);
}

int FillPropertiesNode::getHatchStyle() const
{
	return getHatchStyleField()->getValue();
}

////////////////////////////////////////////////
//	HatchColor
////////////////////////////////////////////////

SFColor *FillPropertiesNode::getHatchColorField() const
{
	if (isInstanceNode() == false)
		return hatchColorField;
	return (SFColor *)getExposedField(hatchColorFieldString);
}

void FillPropertiesNode::setHatchColor(float value[]) 
{
	getHatchColorField()->setValue(value);
}

void FillPropertiesNode::setHatchColor(float r, float g, float b) 
{
	getHatchColorField()->setValue(r, g, b);
}

void FillPropertiesNode::getHatchColor(float value[]) const
{
	getHatchColorField()->getValue(value);
}

////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

FillPropertiesNode *FillPropertiesNode::next() const
{
	return (FillPropertiesNode *)Node::next(getType());
}

FillPropertiesNode *FillPropertiesNode::nextTraversal() const
{
	return (FillPropertiesNode *)Node::nextTraversalByType(getType());
}

////////////////////////////////////////////////
//	functions
////////////////////////////////////////////////
	
bool FillPropertiesNode::isChildNodeType(Node *node) const
{
	return false;
}

void FillPropertiesNode::initialize() 
{
}

void FillPropertiesNode::uninitialize() 
{
}

void FillPropertiesNode::update() 
{
}

////////////////////////////////////////////////
//	Infomation
////////////////////////////////////////////////

void FillPropertiesNode::outputContext(std::ostream &printStream, const char *indentString) const
{
}
