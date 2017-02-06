/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	MFBool.h
*
******************************************************************/

#ifndef _CX3D_MFBOOL_H_
#define _CX3D_MFBOOL_H_

#include <x3d/MField.h>
#include <x3d/SFBool.h>

namespace CyberX3D {

class MFBool : public MField {

	static	int	mInit;

public:

	MFBool();

	void addValue(bool value);
	void addValue(SFBool *sfvalue);
	void addValue(const char *value);

	void insertValue(int index, bool value);
	bool get1Value(int index) const;
	void set1Value(int index, bool value);

	void setValue(MField *mfield);
	void setValue(MFBool *values);
	void setValue(int size, bool values[]);

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
