/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	InterpolatorNode.h
*
******************************************************************/

#ifndef _CX3D_INTERPOLATOR_H_
#define _CX3D_INTERPOLATOR_H_

#include <x3d/Node.h>

namespace CyberX3D {

class InterpolatorNode : public Node {

	MFFloat *keyField;
	SFFloat *fractionField;

public:

	InterpolatorNode();
	virtual ~InterpolatorNode();

	////////////////////////////////////////////////
	//	key
	////////////////////////////////////////////////

	MFFloat *getKeyField() const;
	
	void addKey(float value);
	int getNKeys() const;
	float getKey(int index) const;

	////////////////////////////////////////////////
	//	fraction
	////////////////////////////////////////////////
	
	SFFloat *getFractionField() const;

	void setFraction(float value);
	float getFraction() const;
};

}

#endif
