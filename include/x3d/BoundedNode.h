/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	BoundedNode.h
*
******************************************************************/

#ifndef _CX3D_BOUNDEDNODE_H_
#define _CX3D_BOUNDEDNODE_H_

#include <x3d/Node.h>
#include <x3d/BoundedObject.h>
#include <x3d/BoundingBox.h>

namespace CyberX3D {

class BoundedNode : public Node, public BoundedObject {

	SFVec3f *bboxCenterField;
	SFVec3f *bboxSizeField;

public:

	BoundedNode();
	virtual ~BoundedNode();

	////////////////////////////////////////////////
	//	BoundingBoxSize
	////////////////////////////////////////////////

	SFVec3f *getBoundingBoxSizeField() const;

	void setBoundingBoxSize(float value[]);
	void setBoundingBoxSize(float x, float y, float z);
	void getBoundingBoxSize(float value[]) const;

	////////////////////////////////////////////////
	//	BoundingBoxCenter
	////////////////////////////////////////////////

	SFVec3f *getBoundingBoxCenterField() const;

	void setBoundingBoxCenter(float value[]);
	void setBoundingBoxCenter(float x, float y, float z);
	void getBoundingBoxCenter(float value[]) const;

	////////////////////////////////////////////////
	//	BoundingBox
	////////////////////////////////////////////////

	void setBoundingBox(BoundingBox *bbox);
	void updateBoundingBox();
};

void UpdateBoundingBox(Node *node, BoundingBox *bbox);

}

#endif

