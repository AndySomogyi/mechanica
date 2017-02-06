/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	KeySensorNode.h
*
******************************************************************/

#ifndef _CX3D_KEYSENSORNODE_H_
#define _CX3D_KEYSENSORNODE_H_

#include <x3d/KeyDeviceSensorNode.h>

namespace CyberX3D {

class KeySensorNode : public KeyDeviceSensorNode 
{

public:

	KeySensorNode();
	virtual ~KeySensorNode();

	////////////////////////////////////////////////
	//	List
	////////////////////////////////////////////////

	KeySensorNode *next() const;
	KeySensorNode *nextTraversal() const;

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

