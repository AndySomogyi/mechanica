/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	MFColorRGBA.h
*
******************************************************************/

#ifndef _CX3D_MFCOLORRGBA_H_
#define _CX3D_MFCOLORRGBA_H_

#include <x3d/MField.h>
#include <x3d/SFColorRGBA.h>

namespace CyberX3D {

class MFColorRGBA : public MField {

	static	int	mInit;

public:

	MFColorRGBA();

	void InitializeJavaIDs();

	void addValue(float r, float g, float b, float a);
	void addValue(float value[]);
	void addValue(SFColorRGBA *color);
	void addValue(const char *value);

	void insertValue(int index, float r, float g, float b, float a);
	void insertValue(int index, float value[]);
	void insertValue(int index, SFColorRGBA *color);
	
	void get1Value(int index, float value[]) const;
	
	void set1Value(int index, float value[]);
	void set1Value(int index, float r, float g, float b, float a);

	void setValue(MField *mfield);
	void setValue(MFColorRGBA *colors);
	void setValue(int size, float colors[][4]);

	int getValueCount() const 
	{
		return 4;
	}

	////////////////////////////////////////////////
	//	Output
	////////////////////////////////////////////////

	void outputContext(std::ostream& printStream, const char *indentString) const;
};

}

#endif
