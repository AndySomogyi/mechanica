/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	Grouping2DNode.h
*
******************************************************************/

#ifndef _CX3D_GROUPING2DNODE_H_
#define _CX3D_GROUPING2DNODE_H_

#include <x3d/Node.h>
#include <x3d/BoundingBox.h>

namespace CyberX3D {

class Grouping2DNode : public Node {

	SFVec2f *bboxCenterField;
	SFVec2f *bboxSizeField;

public:

	Grouping2DNode();
	virtual ~Grouping2DNode();

	////////////////////////////////////////////////
	//	BoundingBoxSize
	////////////////////////////////////////////////

	SFVec2f *getBoundingBoxSizeField() const;

	void setBoundingBoxSize(float value[]);
	void setBoundingBoxSize(float x, float y);
	void getBoundingBoxSize(float value[]) const;

	////////////////////////////////////////////////
	//	BoundingBoxCenter
	////////////////////////////////////////////////

	SFVec2f *getBoundingBoxCenterField() const;

	void setBoundingBoxCenter(float value[]);
	void setBoundingBoxCenter(float x, float y);
	void getBoundingBoxCenter(float value[]) const;

	////////////////////////////////////////////////
	//	List
	////////////////////////////////////////////////

	Grouping2DNode *next() const;
	Grouping2DNode *nextTraversal() const;
};

}

#endif

