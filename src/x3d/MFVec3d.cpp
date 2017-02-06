/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	MFVec3d.cpp
*
******************************************************************/

#include <x3d/MFVec3d.h>

using namespace CyberX3D;

MFVec3d::MFVec3d() 
{
	setType(fieldTypeMFVec3d);
}

void MFVec3d::addValue(double x, double y, double z) 
{
	SFVec3d *vector = new SFVec3d(x, y, z);
	add(vector);
}

void MFVec3d::addValue(double value[]) 
{
	SFVec3d *vector = new SFVec3d(value);
	add(vector);
}

void MFVec3d::addValue(SFVec3d *vector) 
{
	add(vector);
}

void MFVec3d::addValue(const char *value) 
{
	SFVec3d *field = new SFVec3d();
	field->setValue(value);
	add(field);
}

void MFVec3d::insertValue(int index, double x, double y, double z) 
{
	SFVec3d *vector = new SFVec3d(x, y, z);
	insert(vector, index);
}

void MFVec3d::insertValue(int index, double value[]) 
{
	SFVec3d *vector = new SFVec3d(value);
	insert(vector, index);
}

void MFVec3d::insertValue(int index, SFVec3d *vector) 
{
	insert(vector, index);
}

void MFVec3d::get1Value(int index, double value[]) const
{
	const SFVec3d *vector = (SFVec3d *)getObject(index);
	if (vector)
		vector->getValue(value);
}

void MFVec3d::set1Value(int index, double value[]) 
{
	SFVec3d *vector = (SFVec3d *)getObject(index);
	if (vector)
		vector->setValue(value);
}

void MFVec3d::set1Value(int index, double x, double y, double z) 
{
	SFVec3d *vector = (SFVec3d *)getObject(index);
	if (vector)
		vector->setValue(x, y, z);
}

void MFVec3d::setValue(MFVec3d *vectors)
{
	clear();

	double value[3];
	int size = vectors->getSize();
	for (int n=0; n<size; n++) {
		vectors->get1Value(n, value);
		addValue(value);
	}
}

void MFVec3d::setValue(MField *mfield)
{
	if (mfield->getType() == fieldTypeMFVec3d)
		setValue((MFVec3d *)mfield);
}

void MFVec3d::setValue(int size, double vectors[][3])
{
	clear();

	for (int n=0; n<size; n++)
		addValue(vectors[n]);
}

////////////////////////////////////////////////
//	Output
////////////////////////////////////////////////

void MFVec3d::outputContext(std::ostream& printStream, const char *indentString) const
{
	double value[3];
	for (int n=0; n<getSize(); n++) {
		get1Value(n, value);
		if (n < getSize()-1)
			printStream << indentString << value[0] << " " << value[1] << " " << value[2] << "," << std::endl;
		else	
			printStream << indentString << value[0] << " " << value[1] << " " << value[2] << std::endl;
	}
}
