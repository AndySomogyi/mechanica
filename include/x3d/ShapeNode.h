/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File: ShapeNode.h
*
*	Revisions:
*
*	12/02/02
*		- Changed the super class from Node to BoundedNode.
*		- Added the follwing new X3D fields.
*			appearance, geometry
*
******************************************************************/

#ifndef _CX3D_SHAPENODE_H_
#define _CX3D_SHAPENODE_H_

#include <x3d/VRML97Fields.h>
#include <x3d/BoundedNode.h>
#include <x3d/AppearanceNode.h>
#include <x3d/Geometry3DNode.h>

namespace CyberX3D {

class ShapeNode : public BoundedNode {

	SFNode *appField;
	SFNode *geomField;

public:

	ShapeNode();
	virtual ~ShapeNode();

	////////////////////////////////////////////////
	//	Appearance
	////////////////////////////////////////////////

	SFNode *getAppearanceField() const;

	////////////////////////////////////////////////
	//	Geometry
	////////////////////////////////////////////////

	SFNode *getGeometryField() const;

	////////////////////////////////////////////////
	//	List
	////////////////////////////////////////////////

	ShapeNode *next() const;
	ShapeNode *nextTraversal() const;

	////////////////////////////////////////////////
	//	Geometry
	////////////////////////////////////////////////

	Geometry3DNode *getGeometry3D() const;

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

