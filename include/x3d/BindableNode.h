/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	BindableNode.h
*
******************************************************************/

#ifndef _CX3D_BINDABLENODE_H_
#define _CX3D_BINDABLENODE_H_

#include <x3d/VRML97Fields.h>
#include <x3d/Node.h>

namespace CyberX3D {

class BindableNode : public Node {

	SFBool *setBindField;
	SFTime *bindTimeField;
	SFBool *isBoundField;

public:

	BindableNode();
	virtual ~BindableNode();

	////////////////////////////////////////////////
	//	bind
	////////////////////////////////////////////////

	SFBool *getBindField() const;

	void setBind(bool value);
	bool  getBind() const;
	bool  isBind() const;

	////////////////////////////////////////////////
	//	bindTime
	////////////////////////////////////////////////
	
	SFTime *getBindTimeField() const;

	void setBindTime(double value);
	double getBindTime() const;

	////////////////////////////////////////////////
	//	isBound
	////////////////////////////////////////////////

	SFBool *getIsBoundField() const;

	void setIsBound(bool  value);
	bool  getIsBound() const;
	bool  isBound() const;
};

}

#endif
