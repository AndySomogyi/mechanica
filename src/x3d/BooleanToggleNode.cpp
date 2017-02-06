/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	BooleanToggleNode.cpp
*
*	Revisions;
*
*	01/08/03
*		- first revision
*
******************************************************************/

#include <x3d/BooleanToggleNode.h>

using namespace CyberX3D;

static const char set_booleanFieldString[] = "set_boolean";
static const char toggleFieldString[] = "toggle";

BooleanToggleNode::BooleanToggleNode() 
{
	setHeaderFlag(false);
	setType(BOOLEANTOGGLE_NODE);

	// set_booleanEvent eventIn field
	set_booleanField = new SFBool();
	addEventIn(set_booleanFieldString, set_booleanField);

	// toggle eventIn field
	toggleField = new SFBool(false);
	addExposedField(toggleFieldString, toggleField);
}

BooleanToggleNode::~BooleanToggleNode() 
{
}

////////////////////////////////////////////////
//	setBoolean
////////////////////////////////////////////////

SFBool *BooleanToggleNode::getBooleanField() const
{
	if (isInstanceNode() == false)
		return set_booleanField;
	return (SFBool*)getEventIn(set_booleanFieldString);
}
	
void BooleanToggleNode::setBoolean(bool value) 
{
	getBooleanField()->setValue(value);
}

bool BooleanToggleNode::getBoolean()  const
{
	return getBooleanField()->getValue();
}

bool BooleanToggleNode::isBoolean()  const
{
	return getBooleanField()->getValue();
}

////////////////////////////////////////////////
//	toggle
////////////////////////////////////////////////

SFBool *BooleanToggleNode::getToggleField() const
{
	if (isInstanceNode() == false)
		return toggleField;
	return (SFBool*)getExposedField(toggleFieldString);
}
	
void BooleanToggleNode::setToggle(bool value) 
{
	getToggleField()->setValue(value);
}

bool BooleanToggleNode::getToggle()  const
{
	return getToggleField()->getValue();
}

bool BooleanToggleNode::isToggle()  const
{
	return getToggleField()->getValue();
}

////////////////////////////////////////////////
//	functions
////////////////////////////////////////////////
	
bool BooleanToggleNode::isChildNodeType(Node *node) const
{
	return false;
}

void BooleanToggleNode::initialize() 
{
}

void BooleanToggleNode::uninitialize() 
{
}

void BooleanToggleNode::update() 
{
}

////////////////////////////////////////////////
//	Output
////////////////////////////////////////////////

void BooleanToggleNode::outputContext(std::ostream &printStream, const char *indentString)  const
{
}

////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

BooleanToggleNode *BooleanToggleNode::next()  const
{
	return (BooleanToggleNode *)Node::next(getType());
}

BooleanToggleNode *BooleanToggleNode::nextTraversal()  const
{
	return (BooleanToggleNode *)Node::nextTraversalByType(getType());
}

