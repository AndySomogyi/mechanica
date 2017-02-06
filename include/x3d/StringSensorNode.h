/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	StringSensorNode.h
*
******************************************************************/

#ifndef _CX3D_STRINGSENSORNODE_H_
#define _CX3D_STRINGSENSORNODE_H_

#include <x3d/KeyDeviceSensorNode.h>

namespace CyberX3D {

class StringSensorNode : public KeyDeviceSensorNode 
{
	SFInt32 *deletionCharacterField;
	SFString *enteredTextField;
	SFString *finalTextField;
	SFInt32 *profileField;
	SFString *terminationTextField;

public:

	StringSensorNode();
	virtual ~StringSensorNode();

	////////////////////////////////////////////////
	//	DeletionCharacter
	////////////////////////////////////////////////

	SFInt32 *getDeletionCharacterField() const;
	
	void setDeletionCharacter(int value);
	int getDeletionCharacter() const;

	////////////////////////////////////////////////
	//	Profile
	////////////////////////////////////////////////

	SFInt32 *getProfileField() const;
	
	void setProfile(int value);
	int getProfile() const;

	////////////////////////////////////////////////
	//	EnteredText
	////////////////////////////////////////////////
	
	SFString *getEnteredTextField() const;

	void setEnteredText(const char *value);
	const char *getEnteredText() const;

	////////////////////////////////////////////////
	//	FinalText
	////////////////////////////////////////////////
	
	SFString *getFinalTextField() const;

	void setFinalText(const char *value);
	const char *getFinalText() const;

	////////////////////////////////////////////////
	//	TerminationText
	////////////////////////////////////////////////
	
	SFString *getTerminationTextField() const;

	void setTerminationText(const char *value);
	const char *getTerminationText() const;

	////////////////////////////////////////////////
	//	List
	////////////////////////////////////////////////

	StringSensorNode *next() const;
	StringSensorNode *nextTraversal() const;

	////////////////////////////////////////////////
	//	functions
	////////////////////////////////////////////////
	
	bool isChildNodeType(Node *node) const;
	void initialize();
	void uninitialize();
	void update();

	////////////////////////////////////////////////
	//	Infomation
	////////////////////////////////////////////////

	void outputContext(std::ostream &printStream, const char *indentString) const;
};

}

#endif

