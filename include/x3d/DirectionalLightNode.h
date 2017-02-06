/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	DirectionalLightNode.h
*
******************************************************************/

#ifndef _CX3D_DIRECTIONALLIGHTNODE_H_
#define _CX3D_DIRECTIONALLIGHTNODE_H_

#include <x3d/VRML97Fields.h>
#include <x3d/LightNode.h>

namespace CyberX3D {

class DirectionalLightNode : public LightNode {

	SFVec3f *directionField;
	
public:

	DirectionalLightNode();
	virtual ~DirectionalLightNode();

	////////////////////////////////////////////////
	//	Direction
	////////////////////////////////////////////////

	SFVec3f *getDirectionField() const;

	void setDirection(float value[]);
	void setDirection(float x, float y, float z);
	void getDirection(float value[]) const;

	////////////////////////////////////////////////
	//	List
	////////////////////////////////////////////////

	DirectionalLightNode *next() const;
	DirectionalLightNode *nextTraversal() const;

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

