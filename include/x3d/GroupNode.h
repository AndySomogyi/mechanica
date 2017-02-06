/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	GroupNode.h
*
******************************************************************/

#ifndef _CX3D_GROUPNODE_H_
#define _CX3D_GROUPNODE_H_

#include <x3d/BoundedGroupingNode.h>

namespace CyberX3D {

class GroupNode : public BoundedGroupingNode {

public:

	GroupNode();
	virtual ~GroupNode();

	////////////////////////////////////////////////
	//	Output
	////////////////////////////////////////////////

	void outputContext(std::ostream &printStream, const char *indentString) const;

	////////////////////////////////////////////////
	//	functions
	////////////////////////////////////////////////
	
	bool isChildNodeType(Node *node) const;
	void initialize();
	void uninitialize();
	void update();

	////////////////////////////////////////////////
	//	List
	////////////////////////////////////////////////

	GroupNode *next() const;
	GroupNode *nextTraversal() const;

};

}

#endif
