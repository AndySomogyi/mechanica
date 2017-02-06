/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	RouteNode.h
*
******************************************************************/

#ifndef _CX3D_ROOTNODE_H_
#define _CX3D_ROOTNODE_H_

#include <x3d/VRML97Fields.h>
#include <x3d/Node.h>

namespace CyberX3D {

class RootNode : public Node {

public:

	RootNode();
	virtual ~RootNode();

	////////////////////////////////////////////////
	//	functions
	////////////////////////////////////////////////
	
	bool isChildNodeType(Node *node) const;
	void initialize();
	void uninitialize();
	void update();

	////////////////////////////////////////////////
	//	infomation
	////////////////////////////////////////////////

	void outputContext(std::ostream& printStream, const char * indentString) const;
};

}

#endif
