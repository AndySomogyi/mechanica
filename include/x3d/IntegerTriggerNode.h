/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	IntegerTriggerNode.h
*
******************************************************************/

#ifndef _CX3D_INTEGERTRIGGERNODE_H_
#define _CX3D_INTEGERTRIGGERNODE_H_

#include <x3d/VRML97Fields.h>
#include <x3d/TriggerNode.h>

namespace CyberX3D {

class IntegerTriggerNode : public TriggerNode {
	
	SFBool *set_booleanField;
	SFInt32 *integerKeyField;
	SFInt32 *triggerValueField;

public:

	IntegerTriggerNode();
	virtual ~IntegerTriggerNode();

	////////////////////////////////////////////////
	//	Boolean
	////////////////////////////////////////////////

	SFBool *getBooleanField() const;
	
	void setBoolean(bool value);
	bool getBoolean() const;
	bool isBoolean() const;

	////////////////////////////////////////////////
	//	IntegerKey
	////////////////////////////////////////////////

	SFInt32* getIntegerKeyField() const;
	
	void setIntegerKey(int value);
	int getIntegerKey() const;

	////////////////////////////////////////////////
	//	triggerValue
	////////////////////////////////////////////////

	SFInt32* getTriggerValueField() const;
	
	void setTriggerValue(int value);
	int getTriggerValue() const;

	////////////////////////////////////////////////
	//	functions
	////////////////////////////////////////////////
	
	bool isChildNodeType(Node *node) const;
	void initialize();
	void uninitialize();
	void update();

	////////////////////////////////////////////////
	//	Output
	////////////////////////////////////////////////

	void outputContext(std::ostream &printStream, const char *indentString) const;

	////////////////////////////////////////////////
	//	List
	////////////////////////////////////////////////

	IntegerTriggerNode *next() const;
	IntegerTriggerNode *nextTraversal() const;

};

}

#endif

