/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	MFVec2d.cpp
*
******************************************************************/

#include <x3d/MFVec2d.h>

using namespace CyberX3D;

MFVec2d::MFVec2d() 
{
	setType(fieldTypeMFVec2d);
}

void MFVec2d::addValue(double x, double y) 
{
	SFVec2d *vector = new SFVec2d(x, y);
	add(vector);
}

void MFVec2d::addValue(double value[]) 
{
	SFVec2d *vector = new SFVec2d(value);
	add(vector);
}

void MFVec2d::addValue(SFVec2d *vector) 
{
	add(vector);
}

void MFVec2d::addValue(const char *value) 
{
	SFVec2d *field = new SFVec2d();
	field->setValue(value);
	add(field);
}

void MFVec2d::insertValue(int index, double x, double y) 
{
	SFVec2d *vector = new SFVec2d(x, y);
	insert(vector, index);
}

void MFVec2d::insertValue(int index, double value[]) 
{
	SFVec2d *vector = new SFVec2d(value);
	insert(vector, index);
}

void MFVec2d::insertValue(int index, SFVec2d *vector) 
{
	insert(vector, index);
}

void MFVec2d::get1Value(int index, double value[]) const
{
	SFVec2d *vector = (SFVec2d *)getObject(index);
	if (vector)
		vector->getValue(value);
}

void MFVec2d::set1Value(int index, double value[]) 
{
	SFVec2d *vector = (SFVec2d *)getObject(index);
	if (vector)
		vector->setValue(value);
}

void MFVec2d::set1Value(int index, double x, double y) 
{
	SFVec2d *vector = (SFVec2d *)getObject(index);
	if (vector)
		vector->setValue(x, y);
}

void MFVec2d::setValue(MFVec2d *vectors)
{
	clear();

	double value[3];
	int size = vectors->getSize();
	for (int n=0; n<size; n++) {
		vectors->get1Value(n, value);
		addValue(value);
	}
}

void MFVec2d::setValue(MField *mfield)
{
	if (mfield->getType() == fieldTypeMFVec2d)
		setValue((MFVec2d *)mfield);
}

void MFVec2d::setValue(int size, double vectors[][2])
{
	clear();

	for (int n=0; n<size; n++)
		addValue(vectors[n]);
}

////////////////////////////////////////////////
//	Output
////////////////////////////////////////////////

void MFVec2d::outputContext(std::ostream& printStream, const char *indentString) const
{
	double value[2];
	for (int n=0; n<getSize(); n++) {
		get1Value(n, value);
		if (n < getSize()-1)
			printStream << indentString << value[0] << " " << value[1] << "," << std::endl;
		else	
			printStream << indentString << value[0] << " " << value[1] << std::endl;
	}
}
