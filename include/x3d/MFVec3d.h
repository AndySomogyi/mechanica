/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	MFVec3d.h
*
******************************************************************/

#ifndef _CX3D_MFVEC3D_H_
#define _CX3D_MFVEC3D_H_

#include <x3d/MField.h>
#include <x3d/SFVec3d.h>

namespace CyberX3D {

class MFVec3d : public MField {

public:

	MFVec3d();

	void addValue(double x, double y, double z);
	void addValue(double value[]);
	void addValue(SFVec3d *vector);
	void addValue(const char *value);

	void insertValue(int index, double x, double y, double z);
	void insertValue(int index, double value[]);
	void insertValue(int index, SFVec3d *vector);

	void get1Value(int index, double value[]) const;
	void set1Value(int index, double value[]);
	void set1Value(int index, double x, double y, double z);

	void setValue(MField *mfield);
	void setValue(MFVec3d *vectors);
	void setValue(int size, double vectors[][3]);

	int getValueCount() const 
	{
		return 3;
	}

	////////////////////////////////////////////////
	//	Output
	////////////////////////////////////////////////

	void outputContext(std::ostream& printStream, const char *indentString) const;

};

}

#endif
