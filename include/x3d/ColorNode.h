/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	ColorNode.h
*
******************************************************************/

#ifndef _CX3D_COLORNODE_H_
#define _CX3D_COLORNODE_H_

#include <x3d/GeometricPropertyNode.h>

namespace CyberX3D {

class ColorNode : public GeometricPropertyNode 
{

	MFColor *colorField;

public:

	ColorNode();
	virtual ~ColorNode();

	////////////////////////////////////////////////
	//	color
	////////////////////////////////////////////////
	
	MFColor *getColorField() const;

	void addColor(float color[]);
	int getNColors() const;
	void getColor(int index, float color[]) const;

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

	ColorNode *next() const;
	ColorNode *nextTraversal() const;
};

}

#endif
