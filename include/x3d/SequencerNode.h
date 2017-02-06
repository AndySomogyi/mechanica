/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	SequencerNode.h
*
******************************************************************/

#ifndef _CX3D_SEQUENCERNODE_H_
#define _CX3D_SEQUENCERNODE_H_

#include <x3d/Node.h>

namespace CyberX3D {

class SequencerNode : public Node {

	MFFloat *keyField;
	SFFloat *fractionField;

public:

	SequencerNode();
	virtual ~SequencerNode();

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
