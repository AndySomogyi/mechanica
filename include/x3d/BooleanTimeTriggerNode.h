/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	BooleanTimeTriggerNode.h
*
******************************************************************/

#ifndef _CX3D_BOOLEANTIMETRIGGERNODE_H_
#define _CX3D_BOOLEANTIMETRIGGERNODE_H_

#include <x3d/VRML97Fields.h>
#include <x3d/TriggerNode.h>

namespace CyberX3D {

class BooleanTimeTriggerNode : public TriggerNode {
	
	SFBool *set_booleanTrueField;
	SFBool *set_booleanFalseField;
	SFBool *trueTriggerField;
	SFBool *falseTriggerField;

public:

	BooleanTimeTriggerNode();
	virtual ~BooleanTimeTriggerNode();

	////////////////////////////////////////////////
	//	SetBooleanTrue
	////////////////////////////////////////////////

	SFBool *getSetBooleanTrueField() const;
	
	void setSetBooleanTrue(bool value);
	bool getSetBooleanTrue() const;

	////////////////////////////////////////////////
	//	SetBooleanFalse
	////////////////////////////////////////////////

	SFBool *getSetBooleanFalseField() const;
	
	void setSetBooleanFalse(bool value);
	bool getSetBooleanFalse() const;

	////////////////////////////////////////////////
	//	TrueTrigger
	////////////////////////////////////////////////

	SFBool *getTrueTriggerField() const;
	
	void setTrueTrigger(bool value);
	bool getTrueTrigger() const;

	////////////////////////////////////////////////
	//	FalseTrigger
	////////////////////////////////////////////////

	SFBool *getFalseTriggerField() const;
	
	void setFalseTrigger(bool value);
	bool getFalseTrigger() const;

	////////////////////////////////////////////////
	//	List
	////////////////////////////////////////////////

	BooleanTimeTriggerNode *next() const;
	BooleanTimeTriggerNode *nextTraversal() const;

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

