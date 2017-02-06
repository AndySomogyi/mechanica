/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	DEF.h
*
******************************************************************/

#ifndef _CX3D_DEF_H_
#define _CX3D_DEF_H_

#include <x3d/StringUtil.h>
#include <x3d/LinkedListNode.h>

namespace CyberX3D {

class DEF : public LinkedListNode<DEF> {

	String		mName;
	String		mString;

public:

	DEF (const char *name, const char *string);
	virtual ~DEF();

	////////////////////////////////////////////////
	//	Name
	////////////////////////////////////////////////

	void setName(const char *name);
	const char *getName() const;

	////////////////////////////////////////////////
	//	Name
	////////////////////////////////////////////////

	void setString(const char *string);
	const char *getString() const;
};

}

#endif


