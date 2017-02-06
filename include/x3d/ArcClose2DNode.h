/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	ArcClose2DNode.h
*
******************************************************************/

#ifndef _CX3D_ARCCLOSE2DNODE_H_
#define _CX3D_ARCCLOSE2DNODE_H_

#include <x3d/Arc2DNode.h>

namespace CyberX3D {

class ArcClose2DNode : public Arc2DNode {

	SFString *closureTypeField;

public:

	ArcClose2DNode();
	virtual ~ArcClose2DNode();

	////////////////////////////////////////////////
	//	ClosureType
	////////////////////////////////////////////////

	SFString *getClosureTypeField() const;
	
	void setClosureType(const char *value);
	const char *getClosureType() const;

	////////////////////////////////////////////////
	//	List
	////////////////////////////////////////////////

	ArcClose2DNode *next() const;
	ArcClose2DNode *nextTraversal() const;

	////////////////////////////////////////////////
	//	functions
	////////////////////////////////////////////////
	
	bool isChildNodeType(Node *node) const;
	void initialize();
	void uninitialize();
	void update();

	////////////////////////////////////////////////
	//	recomputeDisplayList
	////////////////////////////////////////////////

#ifdef CX3D_SUPPORT_OPENGL
	void recomputeDisplayList();
#endif

	////////////////////////////////////////////////
	//	Infomation
	////////////////////////////////////////////////

	void outputContext(std::ostream &printStream, const char *indentString) const;
};

}

#endif
