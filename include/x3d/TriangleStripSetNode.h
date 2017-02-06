/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	TriangleStripSetNode.h
*
******************************************************************/

#ifndef _CX3D_TRIANGLESTRIPSETNODE_H_
#define _CX3D_TRIANGLESTRIPSETNODE_H_

#include <x3d/TriangleSetNode.h>

namespace CyberX3D {

class TriangleStripSetNode : public TriangleSetNode {

	MFInt32 *stripCountField;
	
public:

	TriangleStripSetNode();
	virtual ~TriangleStripSetNode();
	
	////////////////////////////////////////////////
	// StripCount
	////////////////////////////////////////////////

	MFInt32 *getStripCountField() const;

	void addStripCount(int value);
	int getNStripCountes() const;
	int getStripCount(int index) const;

	////////////////////////////////////////////////
	//	List
	////////////////////////////////////////////////

	TriangleStripSetNode *next() const;
	TriangleStripSetNode *nextTraversal() const;

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

