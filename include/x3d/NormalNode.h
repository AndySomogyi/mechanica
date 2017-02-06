/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	NormalNode.h
*
******************************************************************/

#ifndef _CX3D_NORMALNODE_H_
#define _CX3D_NORMALNODE_H_

#include <x3d/VRML97Fields.h>
#include <x3d/GeometricPropertyNode.h>

namespace CyberX3D {

class NormalNode : public GeometricPropertyNode
{

	MFVec3f *vectorField;

public:

	NormalNode();
	virtual ~NormalNode();

	////////////////////////////////////////////////
	//	vector
	////////////////////////////////////////////////
	
	MFVec3f *getVectorField() const;

	void addVector(float value[]);
	int getNVectors() const;
	void getVector(int index, float value[]) const;

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

	NormalNode *next() const;
	NormalNode *nextTraversal() const;
};

}

#endif

