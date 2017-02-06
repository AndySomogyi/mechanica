/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	DEF.cpp
*
******************************************************************/

#include <x3d/DEF.h>

using namespace CyberX3D;

DEF::DEF(const char *name, const char *string) 
{
	setName(name);
	setString(string);
}

DEF::~DEF() 
{
	remove();
}

////////////////////////////////////////////////
//	Name
////////////////////////////////////////////////

void DEF::setName(const char *name) 
{
	mName.setValue(name);
}

const char *DEF::getName() const
{
	return mName.getValue();
}

////////////////////////////////////////////////
//	String
////////////////////////////////////////////////

void DEF::setString(const char *string) 
{
	mString.setValue(string);
}

const char *DEF::getString() const
{
	return mString.getValue();
}
