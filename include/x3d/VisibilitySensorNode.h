/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	VisibilitySensorNode.h
*
******************************************************************/

#ifndef _CX3D_VISIBILITYSENSORNODE_H_
#define _CX3D_VISIBILITYSENSORNODE_H_

#include <x3d/EnvironmentalSensorNode.h>

namespace CyberX3D {

class VisibilitySensorNode : public EnvironmentalSensorNode 
{

public:

	VisibilitySensorNode();
	virtual ~VisibilitySensorNode();

	////////////////////////////////////////////////
	//	List
	////////////////////////////////////////////////

	VisibilitySensorNode *next() const;
	VisibilitySensorNode *nextTraversal() const;

	////////////////////////////////////////////////
	//	functions
	////////////////////////////////////////////////
	
	bool	isChildNodeType(Node *node) const;
	void	initialize();
	void	uninitialize();
	void	update();

	////////////////////////////////////////////////
	//	Infomation
	////////////////////////////////////////////////

	void	outputContext(std::ostream &printStream, const char *indentString) const;
};

}

#endif

