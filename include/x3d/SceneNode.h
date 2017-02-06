/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	GroupNode.h
*
*	06/06/03
*		- The first release
*
******************************************************************/

#ifndef _CX3D_SCENENODE_H_
#define _CX3D_SCENENODE_H_

#include <x3d/Node.h>

namespace CyberX3D {

class SceneNode : public Node {

public:

	SceneNode();
	virtual ~SceneNode();

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

	SceneNode *next() const;
	SceneNode *nextTraversal() const;

	////////////////////////////////////////////////
	//	Output
	////////////////////////////////////////////////

	void outputContext(std::ostream &printStream, const char *indentString) const;
};

}

#endif
