/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	SFDouble.h
*
******************************************************************/

#ifndef _CX3D_SFDOUBLE_H_
#define _CX3D_SFDOUBLE_H_

#include <iostream>
#include <stdio.h>
#include <x3d/Field.h>

namespace CyberX3D {

class SFDouble : public Field {

	double mValue; 

public:

	SFDouble();
	SFDouble(double value);
	SFDouble(SFDouble *value);

	void setValue(double value);
	void setValue(SFDouble *fvalue);
	double getValue() const;

	////////////////////////////////////////////////
	//	Output
	////////////////////////////////////////////////

	friend std::ostream& operator<<(std::ostream &s, const SFDouble &value);
	friend std::ostream& operator<<(std::ostream &s, const SFDouble *value);

	////////////////////////////////////////////////
	//	String
	////////////////////////////////////////////////

	void setValue(const char *value);
	const char *getValue(char *buffer, int bufferLen) const;

	int getValueCount() const 
	{
		return 1;
	}

	////////////////////////////////////////////////
	//	Compare
	////////////////////////////////////////////////

	bool equals(Field *field) const;

};

}

#endif
