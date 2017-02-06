/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	BoundedGrouping2DNode.h
*
******************************************************************/

#ifndef _CX3D_BOUNDEDGROUPING2DNODE_H_
#define _CX3D_BOUNDEDGROUPING2DNODE_H_

#include <x3d/Grouping2DNode.h>
#include <x3d/Bounded2DObject.h>
#include <x3d/BoundingBox2D.h>

namespace CyberX3D {

class BoundedGrouping2DNode : public Grouping2DNode, public Bounded2DObject {

	SFVec2f *bboxCenterField;
	SFVec2f *bboxSizeField;

public:

	BoundedGrouping2DNode();
	virtual ~BoundedGrouping2DNode();

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
	//	BoundingBox2D
	////////////////////////////////////////////////

	void setBoundingBox(BoundingBox2D *bbox);
	void updateBoundingBox();
};

void UpdateBoundingBox(Node *node, BoundingBox2D *bbox);

}

#endif

