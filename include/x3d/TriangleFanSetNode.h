/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	TriangleFanSetNode.h
*
******************************************************************/

#ifndef _CX3D_TRIANGLEFANSETNODE_H_
#define _CX3D_TRIANGLEFANSETNODE_H_

#include <x3d/TriangleSetNode.h>

namespace CyberX3D {

class TriangleFanSetNode : public TriangleSetNode {

	MFInt32 *fanCountField;
	
public:

	TriangleFanSetNode();
	virtual ~TriangleFanSetNode();
	
	////////////////////////////////////////////////
	// FanCount
	////////////////////////////////////////////////

	MFInt32 *getFanCountField() const;

	void addFanCount(int value);
	int getNFanCountes() const;
	int getFanCount(int index) const;

	////////////////////////////////////////////////
	//	List
	////////////////////////////////////////////////

	TriangleFanSetNode *next() const;
	TriangleFanSetNode *nextTraversal() const;

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
	//	Polygon
	////////////////////////////////////////////////

	int getNPolygons() const;

	////////////////////////////////////////////////
	//	Infomation
	////////////////////////////////////////////////

	void outputContext(std::ostream &printStream, const char *indentString) const;

};

}

#endif

