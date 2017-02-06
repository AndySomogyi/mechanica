/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	MFChar.h
*
******************************************************************/

#ifndef _CX3D_MFCHAR_H_
#define _CX3D_MFCHAR_H_

#include <x3d/MField.h>
#include <x3d/SFChar.h>

namespace CyberX3D {

class MFChar : public MField {

public:

	MFChar();

	void addValue(char value);
	void addValue(SFChar *sfvalue);
	void addValue(const char *value);

	void insertValue(int index, char value);
	char get1Value(int index) const;
	void set1Value(int index, char value);

	void setValue(MField *mfield);
	void setValue(MFChar *values);
	void setValue(int size, char values[]);

	int getValueCount() const 
	{
		return 1;
	}

	////////////////////////////////////////////////
	//	Output
	////////////////////////////////////////////////

	void outputContext(std::ostream& printStream, const char *indentString) const;

};

}

#endif
