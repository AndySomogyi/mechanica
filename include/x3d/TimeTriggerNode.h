/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	TimeTriggerNode.h
*
******************************************************************/

#ifndef _CX3D_TIMETRIGGERNODE_H_
#define _CX3D_TIMETRIGGERNODE_H_

#include <x3d/VRML97Fields.h>
#include <x3d/TriggerNode.h>

namespace CyberX3D {

class TimeTriggerNode : public TriggerNode {
	
	SFBool *set_booleanField;
	SFTime *triggerTimeField;

public:

	TimeTriggerNode();
	virtual ~TimeTriggerNode();

	////////////////////////////////////////////////
	//	Boolean
	////////////////////////////////////////////////

	SFBool *getBooleanField() const;
	
	void setBoolean(bool value);
	bool getBoolean() const;
	bool isBoolean() const;

	////////////////////////////////////////////////
	//	triggerTime
	////////////////////////////////////////////////

	SFTime* getTriggerTimeField() const;
	
	void setTriggerTime(double value);
	double getTriggerTime() const;

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

	TimeTriggerNode *next() const;
	TimeTriggerNode *nextTraversal() const;

};

}

#endif
