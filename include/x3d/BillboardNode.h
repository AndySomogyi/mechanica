/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	BillboardNode.h
*
******************************************************************/

#ifndef _CX3D_BILLBOARDNODE_H_
#define _CX3D_BILLBOARDNODE_H_

#include <x3d/VRML97Fields.h>
#include <x3d/BoundedGroupingNode.h>

namespace CyberX3D {
	
class BillboardNode : public BoundedGroupingNode {

	SFVec3f *axisOfRotationField;

public:

	BillboardNode();
	virtual ~BillboardNode();

	////////////////////////////////////////////////
	//	axisOfRotation
	////////////////////////////////////////////////

	SFVec3f *getAxisOfRotationField() const;

	void setAxisOfRotation(float value[]);
	void setAxisOfRotation(float x, float y, float z);
	void getAxisOfRotation(float value[]) const;

	////////////////////////////////////////////////
	//	List
	////////////////////////////////////////////////

	BillboardNode *next() const;
	BillboardNode *nextTraversal() const;

	////////////////////////////////////////////////
	//	functions
	////////////////////////////////////////////////
	
	bool isChildNodeType(Node *node) const;
	void initialize();
	void uninitialize();
	void update();

	////////////////////////////////////////////////
	//	actions
	////////////////////////////////////////////////

	void	getBillboardToViewerVector(float vector[3]) const;
	void	getViewerToBillboardVector(float vector[3]) const;
	void	getPlaneVectorOfAxisOfRotationAndBillboardToViewer(float vector[3]) const;
	void	getZAxisVectorOnPlaneOfAxisOfRotationAndBillboardToViewer(float vector[3]) const;
	float	getRotationAngleOfZAxis() const;
	void	getRotationZAxisRotation(float rotation[4]) const;
	void	getSFMatrix(SFMatrix *mOut) const;

	////////////////////////////////////////////////
	//	Infomation
	////////////////////////////////////////////////

	void outputContext(std::ostream &printStream, const char *indentString) const;
};

}

#endif
