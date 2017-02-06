/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	StringSensorNode.cpp
*
******************************************************************/

#include <x3d/StringSensorNode.h>

using namespace CyberX3D;

static const char deletionCharacterFieldString[] = "deletionCharacter";
static const char enteredTextFieldString[] = "enteredText";
static const char finalTextFieldString[] = "finalText";
static const char profileFieldString[] = "profile";
static const char terminationTextFieldString[] = "terminationText";

StringSensorNode::StringSensorNode() 
{
	setHeaderFlag(false);
	setType(STRINGSENSOR_NODE);

	// deletionCharacter eventOut field
	deletionCharacterField = new SFInt32(0);
	addEventOut(deletionCharacterFieldString, deletionCharacterField);

	// profile eventOut field
	profileField = new SFInt32(0);
	addEventOut(profileFieldString, profileField);

	// enteredText field
	enteredTextField = new SFString("");
	addEventOut(enteredTextFieldString, enteredTextField);

	// finalText field
	finalTextField = new SFString("");
	addEventOut(finalTextFieldString, finalTextField);

	// terminationText field
	terminationTextField = new SFString("");
	addEventOut(terminationTextFieldString, terminationTextField);
}

StringSensorNode::~StringSensorNode() 
{
}

////////////////////////////////////////////////
//	DeletionCharacter
////////////////////////////////////////////////

SFInt32 *StringSensorNode::getDeletionCharacterField() const
{
	if (isInstanceNode() == false)
		return deletionCharacterField;
	return (SFInt32 *)getEventOut(deletionCharacterFieldString);
}
	
void StringSensorNode::setDeletionCharacter(int value) 
{
	getDeletionCharacterField()->setValue(value);
}

int StringSensorNode::getDeletionCharacter() const
{
	return getDeletionCharacterField()->getValue();
}

////////////////////////////////////////////////
//	Profile
////////////////////////////////////////////////

SFInt32 *StringSensorNode::getProfileField() const
{
	if (isInstanceNode() == false)
		return profileField;
	return (SFInt32 *)getEventOut(profileFieldString);
}
	
void StringSensorNode::setProfile(int value) 
{
	getProfileField()->setValue(value);
}

int StringSensorNode::getProfile() const
{
	return getProfileField()->getValue();
}

////////////////////////////////////////////////
//	EnteredText
////////////////////////////////////////////////

SFString *StringSensorNode::getEnteredTextField() const
{
	if (isInstanceNode() == false)
		return enteredTextField;
	return (SFString *)getEventOut(enteredTextFieldString);
}
	
void StringSensorNode::setEnteredText(const char *value) 
{
	getEnteredTextField()->setValue(value);
}

const char *StringSensorNode::getEnteredText() const
{
	return getEnteredTextField()->getValue();
}

////////////////////////////////////////////////
//	FinalText
////////////////////////////////////////////////

SFString *StringSensorNode::getFinalTextField() const
{
	if (isInstanceNode() == false)
		return finalTextField;
	return (SFString *)getEventOut(finalTextFieldString);
}
	
void StringSensorNode::setFinalText(const char *value) 
{
	getFinalTextField()->setValue(value);
}

const char *StringSensorNode::getFinalText() const
{
	return getFinalTextField()->getValue();
}

////////////////////////////////////////////////
//	TerminationText
////////////////////////////////////////////////

SFString *StringSensorNode::getTerminationTextField() const
{
	if (isInstanceNode() == false)
		return terminationTextField;
	return (SFString *)getEventOut(terminationTextFieldString);
}
	
void StringSensorNode::setTerminationText(const char *value) 
{
	getTerminationTextField()->setValue(value);
}

const char *StringSensorNode::getTerminationText() const
{
	return getTerminationTextField()->getValue();
}

////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

StringSensorNode *StringSensorNode::next() const
{
	return (StringSensorNode *)Node::next(getType());
}

StringSensorNode *StringSensorNode::nextTraversal() const
{
	return (StringSensorNode *)Node::nextTraversalByType(getType());
}

////////////////////////////////////////////////
//	functions
////////////////////////////////////////////////
	
bool StringSensorNode::isChildNodeType(Node *node) const
{
	return false;
}

void StringSensorNode::initialize() 
{
}

void StringSensorNode::uninitialize() 
{
}

void StringSensorNode::update() 
{
}

////////////////////////////////////////////////
//	Infomation
////////////////////////////////////////////////

void StringSensorNode::outputContext(std::ostream &printStream, const char *indentString) const
{
}
