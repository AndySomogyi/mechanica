/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	BooleanTriggerNode.h
*
******************************************************************/

#ifndef _CX3D_BOOLEANTRIGGERNODE_H_
#define _CX3D_BOOLEANTRIGGERNODE_H_

#include <x3d/VRML97Fields.h>
#include <x3d/TriggerNode.h>

namespace CyberX3D {

class BooleanTriggerNode : public Node {
	
	SFTime *set_triggerTimeField;
	SFBool *triggerTrueField;

public:

	BooleanTriggerNode();
	virtual ~BooleanTriggerNode();

	////////////////////////////////////////////////
	//	TriggerTimeEvent
	////////////////////////////////////////////////

	SFTime *getTriggerTimeField() const;
	
	void setTriggerTime(double value);
	double getTriggerTime() const;

	////////////////////////////////////////////////
	//	IntegerKey
	////////////////////////////////////////////////

	SFBool* getTriggerTrueField() const;
	
	void setTriggerTrue(bool value);
	bool getTriggerTrue() const;
	bool isTriggerTrue() const;

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

	BooleanTriggerNode *next() const;
	BooleanTriggerNode *nextTraversal() const;

};

}

#endif

