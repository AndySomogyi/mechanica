/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	NavigationInfoNode.cpp
*
*	03/25/04
*		- Joerg Scheurich aka MUFTI <rusmufti@helpdesk.rus.uni-stuttgart.de>
*		- Fixed the default value of the headlight field to "true".
*
******************************************************************/

#include <x3d/NavigationInfoNode.h>

using namespace CyberX3D;

NavigationInfoNode::NavigationInfoNode() 
{
	setHeaderFlag(false);
	setType(NAVIGATIONINFO_NODE);

	///////////////////////////
	// Exposed Field 
	///////////////////////////

	// visibilityLimit exposed field
	visibilityLimitField = new SFFloat(0.0f);
	addExposedField(visibilityLimitFieldString, visibilityLimitField);

	// avatarSize exposed field
	avatarSizeField = new MFFloat();
	addExposedField(avatarSizeFieldString, avatarSizeField);

	// type exposed field
	typeField = new MFString();
	addExposedField(typeFieldString, typeField);

	// headlight exposed field
	// Thanks for Joerg Scheurich aka MUFTI (03/25/04)
	headlightField = new SFBool(true);
	addExposedField(headlightFieldString, headlightField);

	// speed exposed field
	speedField = new SFFloat(1.0f);
	addExposedField(speedFieldString, speedField);
}

NavigationInfoNode::~NavigationInfoNode() 
{
}

////////////////////////////////////////////////
// Type
////////////////////////////////////////////////

MFString *NavigationInfoNode::getTypeField() const
{
	if (isInstanceNode() == false)
		return typeField;
	return (MFString *)getExposedField(typeFieldString);
}

void NavigationInfoNode::addType(const char *value) 
{
	getTypeField()->addValue(value);
}

int NavigationInfoNode::getNTypes() const
{
	return getTypeField()->getSize();
}

const char *NavigationInfoNode::getType(int index) const
{
	return getTypeField()->get1Value(index);
}

////////////////////////////////////////////////
// avatarSize
////////////////////////////////////////////////

MFFloat *NavigationInfoNode::getAvatarSizeField() const
{
	if (isInstanceNode() == false)
		return avatarSizeField;
	return (MFFloat *)getExposedField(avatarSizeFieldString);
}

void NavigationInfoNode::addAvatarSize(float value) 
{
	getAvatarSizeField()->addValue(value);
}

int NavigationInfoNode::getNAvatarSizes() const
{
	return getAvatarSizeField()->getSize();
}

float NavigationInfoNode::getAvatarSize(int index) const
{
	return getAvatarSizeField()->get1Value(index);
}

////////////////////////////////////////////////
//	Headlight
////////////////////////////////////////////////

SFBool *NavigationInfoNode::getHeadlightField() const
{
	if (isInstanceNode() == false)
		return headlightField;
	return (SFBool *)getExposedField(headlightFieldString);
}
	
void NavigationInfoNode::setHeadlight(bool value) 
{
	getHeadlightField()->setValue(value);
}

void NavigationInfoNode::setHeadlight(int value) 
{
	setHeadlight(value ? true : false);
}

bool NavigationInfoNode::getHeadlight() const
{
	return getHeadlightField()->getValue();
}

////////////////////////////////////////////////
//	VisibilityLimit
////////////////////////////////////////////////

SFFloat *NavigationInfoNode::getVisibilityLimitField() const
{
	if (isInstanceNode() == false)
		return visibilityLimitField;
	return (SFFloat *)getExposedField(visibilityLimitFieldString);
}

void NavigationInfoNode::setVisibilityLimit(float value) 
{
	getVisibilityLimitField()->setValue(value);
}

float NavigationInfoNode::getVisibilityLimit() const
{
	return getVisibilityLimitField()->getValue();
}

////////////////////////////////////////////////
//	Speed
////////////////////////////////////////////////

SFFloat *NavigationInfoNode::getSpeedField() const
{
	if (isInstanceNode() == false)
		return speedField;
	return (SFFloat *)getExposedField(speedFieldString);
}
	
void NavigationInfoNode::setSpeed(float value) 
{
	getSpeedField()->setValue(value);
}

float NavigationInfoNode::getSpeed() const
{
	return getSpeedField()->getValue();
}

////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

bool NavigationInfoNode::isChildNodeType(Node *node) const
{
	return false;
}

NavigationInfoNode *NavigationInfoNode::next() const
{
	return (NavigationInfoNode *)Node::next(Node::getType());
}

NavigationInfoNode *NavigationInfoNode::nextTraversal() const
{
	return (NavigationInfoNode *)Node::nextTraversalByType(Node::getType());
}

////////////////////////////////////////////////
//	functions
////////////////////////////////////////////////
	
void NavigationInfoNode::initialize() 
{
}

void NavigationInfoNode::uninitialize() 
{
}

void NavigationInfoNode::update() 
{
}

////////////////////////////////////////////////
//	infomation
////////////////////////////////////////////////

void NavigationInfoNode::outputContext(std::ostream &printStream, const char *indentString) const
{
	SFBool *headlight = getHeadlightField();

	printStream << indentString << "\t" << "visibilityLimit " << getVisibilityLimit() << std::endl;
	printStream << indentString << "\t" << "headlight " << headlight << std::endl;
	printStream << indentString << "\t" << "speed " << getSpeed() << std::endl;

	if (0 < getNTypes()) {
		MFString *type = getTypeField();
		printStream << indentString << "\t" << "type [" << std::endl;
		type->MField::outputContext(printStream, indentString, "\t\t");
		printStream << indentString << "\t" << "]" << std::endl;
	}

	if (0 < getNAvatarSizes()) {
		MFFloat *avatarSize = getAvatarSizeField();
		printStream << indentString << "\t" << "avatarSize [" << std::endl;
		avatarSize->MField::outputContext(printStream, indentString, "\t\t");
		printStream << indentString << "\t" << "]" << std::endl;
	}
}
